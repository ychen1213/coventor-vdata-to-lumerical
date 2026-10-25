"""Shared, SDK-free metadata and process helpers. Python 3.6+ compatible."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET

SCHEMA_VERSION = 1
RESERVED_IDS = frozenset(range(250,256))  # Queried from the installed 9.3 SDK.
UNIT_UM = {'m':1e6,'meter':1e6,'meters':1e6,'cm':1e4,'mm':1e3,
           'um':1.,'µm':1.,'μm':1.,'micron':1.,'microns':1.,'micrometer':1.,
           'nm':1e-3,'nanometer':1e-3,'nanometers':1e-3,
           'a':1e-4,'angstrom':1e-4,'angstroms':1e-4,'å':1e-4}
EXTERNAL_TAGS = ('ModelFile','LayoutFile','ProcessFile','MaterialFile')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(str(temporary),str(path))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def unit_factor(unit):
    key=str(unit).strip().lower()
    if key not in UNIT_UM:
        raise ValueError('Unsupported or missing length unit: {!r}'.format(unit))
    return UNIT_UM[key]


def inspect_vdata(path):
    """Read the declared grid, palette and optional fields; never infer Si ID."""
    path=Path(path).resolve()
    if not path.is_file():
        raise ValueError('Input file does not exist: '+str(path))
    if path.stat().st_size==0:
        raise ValueError('Input VDATA is empty')
    try:
        root=ET.parse(str(path)).getroot()
    except ET.ParseError as exc:
        raise ValueError('Not a readable Coventor VoxelXML file: '+str(exc))
    if root.tag!='VoxelXML':
        raise ValueError('Expected Coventor VoxelXML root, found '+str(root.tag))
    props=root.find('Properties')
    if props is None:
        raise ValueError('Missing VoxelXML/Properties')
    def required(tag):
        element=props.find(tag)
        if element is None:
            raise ValueError('Missing Properties/'+tag)
        return element
    size=required('VoxelSize');count=required('NumVoxels');origin=required('ModelOrigin')
    try:
        spacing=float(size.attrib['Value'])*unit_factor(size.get('Units'))
        dimensions=[int(count.attrib[k]) for k in ('NX','NY','NZ')]
        offset=[float(origin.attrib[k])*unit_factor(origin.get('Units')) for k in ('X','Y','Z')]
    except (KeyError,TypeError) as exc:
        raise ValueError('Incomplete grid metadata: '+str(exc))
    if not math.isfinite(spacing) or spacing<=0 or any(n<=0 for n in dimensions):
        raise ValueError('Voxel spacing and all grid dimensions must be positive')
    if not all(math.isfinite(v) for v in offset):
        raise ValueError('Nonfinite model origin')
    upper=[v+n*spacing for v,n in zip(offset,dimensions)]
    material_node=root.find('Materials')
    if material_node is None:
        raise ValueError('Missing material palette')
    palette=[];seen=set()
    for item in material_node:
        if item.get('MaterialID') is None:
            continue
        identifier=int(item.get('MaterialID'))
        if identifier in seen:
            raise ValueError('Duplicate material ID in palette: '+str(identifier))
        seen.add(identifier)
        name=item.get('name')
        if not name:
            raise ValueError('Material ID {} has no name'.format(identifier))
        color=[int(item.get(c,'128')) for c in ('red','green','blue')]
        if any(c<0 or c>255 for c in color):
            raise ValueError('Material RGB is outside 0..255')
        palette.append(dict(id=identifier,name=name,color_rgb=color,type=item.get('type','Dielectric')))
    data=list(root.iter('VoxelData'))
    if not data or not any((n.text or '').strip() for n in data):
        raise ValueError('Missing or empty VoxelData payload')
    return dict(schema_version=SCHEMA_VERSION,input_vdata=str(path),source_sha256=sha256(path),
                file_bytes=path.stat().st_size,xml_root=root.tag,xml_attributes=dict(root.attrib),
                voxel_size_um=spacing,dimensions_xyz=dimensions,origin_um=offset,bounds_um=[offset,upper],
                original_voxel_unit=size.get('Units'),original_origin_unit=origin.get('Units'),
                materials=palette,has_plane_equations=root.find('PlaneEquations') is not None,
                optional_fields=[n.tag for n in root if n.tag not in ('Properties','Materials','VoxelData')],
                external_references={tag:dict(props.find(tag).attrib) for tag in EXTERNAL_TAGS if props.find(tag) is not None},
                wafer_info={n.tag:dict(n.attrib) for n in root.find('WaferInfo')} if root.find('WaferInfo') is not None else {},
                voxel_count=dimensions[0]*dimensions[1]*dimensions[2])


def stage_standalone(source,destination,metadata=None,recenter=True):
    """Clear provenance paths without rewriting CDATA or encoded geometry.

    Coventor 9.3's importer can crash if an XML library converts the VoxelData
    CDATA section to a plain text node. Only the Properties byte range is edited.
    The imported geometry therefore cannot depend on the original ZAM/VWAF/GDS.
    """
    raw=Path(source).read_bytes()
    original=raw
    metadata=metadata or inspect_vdata(source)
    match=re.search(rb'<Properties(?:\s[^>]*)?>.*?</Properties\s*>',raw,re.S)
    if match is None:
        raise ValueError('Cannot isolate Properties for standalone staging')
    before=match.group(0)
    def strip_reference(tag_match):
        return re.sub(rb'(\b(?:name|top)\s*=\s*)([\x22\x27]).*?\2',
                      lambda m:m.group(1)+b'""',tag_match.group(0),flags=re.S)
    after=re.sub(rb'<(?:ModelFile|LayoutFile|ProcessFile|MaterialFile)\b[^>]*>',strip_reference,before,flags=re.S)
    translation=metadata['origin_um'] if recenter else [0.,0.,0.]
    if recenter:
        def zero_origin(m):
            return re.sub(rb'(\b[XYZ]\s*=\s*)([\x22\x27]).*?\2',lambda a:a.group(1)+b'"0"',m.group(0),flags=re.S)
        after=re.sub(rb'<ModelOrigin\b[^>]*>',zero_origin,after,flags=re.S)
    staged=raw[:match.start()]+after+raw[match.end():]
    aliases={str(m['id']):'/VDATA_ID_'+str(m['id']) for m in metadata['materials'] if m['id'] not in RESERVED_IDS}
    def alias_material(m):
        tag=m.group(0)
        identifier=re.search(rb'\bMaterialID\s*=\s*[\x22\x27](\d+)[\x22\x27]',tag)
        if identifier and identifier.group(1).decode('ascii') in aliases:
            value=aliases[identifier.group(1).decode('ascii')].encode('ascii')
            tag=re.sub(rb'(\bname\s*=\s*)([\x22\x27]).*?\2',lambda a:a.group(1)+b'"'+value+b'"',tag,flags=re.S)
        return tag
    materials_match=re.search(rb'<Materials(?:\s[^>]*)?>.*?</Materials\s*>',staged,re.S)
    if materials_match:
        renamed=re.sub(rb'<material\b[^>]*>',alias_material,materials_match.group(0),flags=re.S)
        staged=staged[:materials_match.start()]+renamed+staged[materials_match.end():]
    # CropBounds is stored in the original model unit, independently of
    # ModelOrigin. It must receive the same translation for native meshing.
    factor=unit_factor(metadata['original_voxel_unit'])
    def crop_transform(m):
        def coordinate(a):
            axis=b'xyz'.index(a.group(1)[0:1].lower())
            value=float(a.group(3))-(translation[axis]/factor)
            return a.group(1)+b'="'+format(value,'.17g').encode('ascii')+b'"'
        return re.sub(rb'\b([xyz](?:Min|Max))\s*=\s*([\x22\x27])(.*?)\2',coordinate,m.group(0))
    crop_present=re.search(rb'<CropBounds\b[^>]*>',staged)
    if crop_present:
        staged=re.sub(rb'<CropBounds\b[^>]*>',crop_transform,staged)
    else:
        # This field is optional in VoxelXML, but the 9.3 mesher needs bounds.
        low=[metadata['origin_um'][i]-translation[i] for i in range(3)]
        high=[metadata['bounds_um'][1][i]-translation[i] for i in range(3)]
        attrs=' '.join('{}{}="{:.17g}"'.format(axis,edge,v/factor)
                       for i,axis in enumerate('xyz') for edge,v in (('Min',low[i]),('Max',high[i])))
        crop=('<CropBounds '+attrs+'/>').encode('ascii')
        if re.search(rb'<WaferInfo(?:\s[^>]*)?>',staged):
            staged=re.sub(rb'(<WaferInfo(?:\s[^>]*)?>)',lambda m:m.group(0)+crop,staged,count=1)
        else:
            staged=staged.replace(b'</Properties>',b'</Properties><WaferInfo>'+crop+b'</WaferInfo>',1)
    # The entire encoded body of every data field is byte-identical. Only
    # provenance paths, the coordinate origin and CropBounds are changed.
    def payloads(blob):
        return re.findall(rb'<(VoxelData|PlaneEquations|DopantData)\b[^>]*>(.*?)</\1\s*>',blob,re.S)
    unchanged=payloads(original)==payloads(staged)
    if not unchanged or not payloads(original):
        raise RuntimeError('Standalone staging unexpectedly changed payload bytes')
    Path(destination).write_bytes(staged)
    return dict(path=str(Path(destination).resolve()),source_sha256=sha256(source),
                staged_sha256=sha256(destination),payload_bytes_unchanged=True,
                external_reference_names_cleared=True,translation_um=translation,
                native_material_names=aliases,
                crop_bounds_added=not bool(crop_present),coordinates_recentered=bool(recenter))


def make_ascii_work_directory():
    """9.3 native file APIs cannot reliably handle Unicode paths."""
    choices=[]
    if os.environ.get('VDATA_CONVERTER_SCRATCH'):
        choices.append(installation_path(os.environ['VDATA_CONVERTER_SCRATCH']))
    choices.append(Path(tempfile.gettempdir())/'Coventor_VDATA_Converter')
    if os.environ.get('PROGRAMDATA'):
        choices.append(installation_path(os.environ['PROGRAMDATA'])/'Coventor_VDATA_Converter')
    choices.extend(root/'Temp'/'Coventor_VDATA_Converter'
                   for root in installation_search_roots() if root.parent==root)
    for parent in choices:
        try:
            str(parent).encode('ascii')
            parent.mkdir(parents=True,exist_ok=True)
            folder=Path(tempfile.mkdtemp(prefix='conversion_',dir=str(parent))).resolve()
            return folder,parent.resolve()
        except (UnicodeEncodeError,OSError):
            continue
    raise ValueError('No writable ASCII-only scratch location is available for the Coventor SDK')


def remove_work_directory(folder,parent):
    folder=Path(folder).resolve();parent=Path(parent).resolve()
    if folder.parent!=parent or not folder.name.startswith('conversion_'):
        raise ValueError('Refusing to remove an unverified native scratch directory')
    shutil.rmtree(str(folder))


def publish_native_geometry(work,output):
    """Copy only portable geometry/reports from the native ASCII workspace."""
    work,output=Path(work),Path(output)
    report_path=work/'native_report.json'
    if report_path.exists():
        report=read_json(report_path)
        report['native_work_directory_was_ascii']=True
        report['manifest']=str(output/'conversion_manifest.json')
        write_json(output/'native_report.json',report)
    manifest_path=work/'conversion_manifest.json'
    if not manifest_path.exists():
        return
    manifest=read_json(manifest_path)
    meshes=output/'meshes';meshes.mkdir(exist_ok=True)
    for item in manifest['materials']:
        source=Path(item['stl_path'])
        target=meshes/source.name
        shutil.copy2(str(source),str(target))
        item['stl_path']=str(target.resolve())
    manifest['native_work_directory_was_ascii']=True
    write_json(output/'conversion_manifest.json',manifest)
    if report_path.exists():
        report['active_materials']=manifest['materials']
        if report.get('staging'):
            report['staging']['native_import_path']=report['staging']['path']
            report['staging']['path']=str(output/'standalone_input.vdata')
        write_json(output/'native_report.json',report)


def discover_semulator(explicit=None):
    """Find an installed SDK without assuming a drive or product version.

    Explicit roots are authoritative. Otherwise environment roots precede a
    bounded scan of common installation directories on local fixed drives.
    Recognizing an SDK is not a claim that every SDK version is compatible.
    """
    candidates=[installation_path(explicit)] if explicit else []
    if not explicit:
        for key in ('VDATA_SEMULATOR_ROOT','SEMULATOR_ROOT','MEMULATOR_ROOT'):
            if os.environ.get(key):
                candidates.append(installation_path(os.environ[key]))
        found=[]
        for base in installation_search_roots():
            for pattern in ('SEMulator3D*','Coventor/SEMulator3D*',
                            'Coventor*/SEMulator3D*','Coventor/SEMulator3D/*'):
                found.extend(safe_installation_glob(base,pattern))
        candidates.extend(sorted(found,key=installation_version_key,reverse=True))
    for root in unique_installation_paths(candidates):
        interpreters=[]
        for pattern in ('runtime/python/*/NT*/python.exe',
                        'runtime/python/*/*/python.exe','runtime/python/*/python.exe',
                        'runtime/python/python.exe'):
            interpreters.extend(safe_installation_glob(root,pattern))
        interpreters=sorted(unique_installation_paths(interpreters),key=installation_version_key,reverse=True)
        if (root/'runtime'/'pyc').is_dir() and interpreters:
            return root.resolve(),interpreters[0].resolve()
    suffix=' at '+str(explicit) if explicit else ''
    raise ValueError('SEMulator3D Python SDK not found'+suffix+
                     '; specify --semulator-root or set SEMULATOR_ROOT')


def installation_path(value):
    return Path(os.path.expandvars(str(value).strip().strip('"'))).expanduser()


def installation_version_key(path):
    """Natural version ordering (10 follows 9), with deterministic text parts."""
    return tuple((1,int(part)) if part.isdigit() else (0,part.lower())
                 for part in re.split(r'(\d+)',str(path)))


def unique_installation_paths(paths):
    result=[];seen=set()
    for path in paths:
        path=installation_path(path)
        key=os.path.normcase(os.path.abspath(str(path)))
        if key not in seen:
            seen.add(key);result.append(path)
    return result


def safe_installation_glob(base,pattern):
    # Do not recursively enumerate user files or traverse network drives.
    try:
        return list(Path(base).glob(pattern)) if Path(base).is_dir() else []
    except OSError:
        return []


def installation_search_roots():
    """Common Windows bases, including non-system local installation drives."""
    bases=[]
    for key in ('ProgramW6432','ProgramFiles','ProgramFiles(x86)'):
        if os.environ.get(key):
            bases.append(installation_path(os.environ[key]))
    if os.name=='nt':
        import ctypes
        kernel=ctypes.windll.kernel32
        kernel.GetDriveTypeW.argtypes=[ctypes.c_wchar_p]
        kernel.GetDriveTypeW.restype=ctypes.c_uint
        mask=kernel.GetLogicalDrives()
        for index in range(26):
            if not mask & (1<<index):
                continue
            drive=Path(chr(ord('A')+index)+':/')
            # DRIVE_FIXED: avoid waking removable media or waiting on shares.
            if kernel.GetDriveTypeW(str(drive))!=3:
                continue
            bases.extend((drive,drive/'Program Files',drive/'Program Files (x86)'))
    return unique_installation_paths(bases)


def native_environment(root,python,output,api_version):
    """Equivalent SDK search paths, without routing user paths through cmd.exe."""
    root=Path(root);python=Path(python)
    env=os.environ.copy()
    for key in ('PYTHONHOME','PYTHON_DOM','PYTHONSTARTUP','PYTHONCASEOK'):
        env.pop(key,None)
    env.update(MEMULATOR_ROOT=str(root),MTI_ROOT=str(root),MEMULATOR_PROJECTDIR=str(output),
               MTI_ARCH='NT40_x64',MTI_PRODUCTNAME='SEMulator3D',PYTHONDONTWRITEBYTECODE='1',
               PYTHONLEGACYWINDOWSSTDIO='1',PYTHONIOENCODING='utf-8',
               VDATA_CONVERTER_API_VERSION=str(api_version))
    env['PYTHONPATH']=os.pathsep.join(str(p) for p in (root/'runtime'/'pyc',root/'bin'/'NT40_x64',root/'apps'/'contrib'/'py'))
    env['PATH']=os.pathsep.join([str(root/'bin'/'NT40_x64'),str(root/'bin'),str(python.parent),
                               str(python.parent/'DLLs'),env.get('PATH','')])
    env['TCL_LIBRARY']=str(python.parent/'tcl'/'tcl8.4')
    env['TK_LIBRARY']=str(python.parent/'tcl'/'tk8.4')
    # Use the installation's standard license path only when no configured
    # server/file is inherited. The license contents are never read.
    license_file=root/'license'/'license.lic'
    if not env.get('LM_LICENSE_FILE') and license_file.exists():
        env['LM_LICENSE_FILE']=str(license_file)
    return env


def run_logged(command,log_path,env=None,cwd=None,timeout=None):
    start=time.monotonic()
    with Path(log_path).open('wb') as log:
        child=subprocess.Popen([str(v) for v in command],stdout=log,stderr=subprocess.STDOUT,
                               env=env,cwd=str(cwd) if cwd else None,shell=False)
        last=start
        try:
            while child.poll() is None:
                now=time.monotonic()
                if timeout and now-start>timeout:
                    child.terminate();child.wait(timeout=30)
                    raise RuntimeError('Stage exceeded {} seconds; see {}'.format(timeout,log_path))
                if now-last>=30:
                    print('Stage running for {:.0f}s; log: {}'.format(now-start,log_path),flush=True)
                    last=now
                time.sleep(.25)
        except BaseException:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=30)
            raise
    return dict(command=[str(v) for v in command],returncode=child.returncode,
                elapsed_seconds=time.monotonic()-start,log=str(log_path))
