"""Native independent-file conversion. Run under the installed Coventor SDK."""
import os
import sys
import time
import traceback
from pathlib import Path
_SEMulator3D_API_VERSION=os.environ.get('VDATA_CONVERTER_API_VERSION','9.3')
from voxelModeler import *
from voxelWafer import *
from common import read_json,write_json,unit_factor,sha256,RESERVED_IDS
from mesh_tools import read_obj,audit_and_write


def convert(job):
    start=time.monotonic();output=Path(job['output_dir']);metadata=job['metadata']
    native=output/'native';native.mkdir(exist_ok=True)
    os.chdir(str(native))
    factor=unit_factor(metadata['original_voxel_unit'])
    unit={1.:'um',.001:'nm',.0001:'A'}.get(factor)
    if unit is None:
        raise ValueError('This installed native SDK supports model units um/nm/angstroms; received '+metadata['original_voxel_unit'])
    resolution=metadata['voxel_size_um']/factor
    report=dict(schema_version=1,status='importing',input_vdata=metadata['input_vdata'],
                source_sha256=metadata['source_sha256'],independent_input=True,
                source_project_loaded=False,process_simulation_run=False,
                staging=job['staging'],model_units=unit,resolution_model_units=resolution,
                api_version=_SEMulator3D_API_VERSION)
    report_path=output/'native_report.json';write_json(report_path,report)
    doc=VoxelDocument(str(native/'converted.zam'),resolution,unit,NO_OVERWRITE)
    selected=[];material_records=[]
    for item in metadata['materials']:
        if item['id'] in RESERVED_IDS:
            continue
        kind=CONDUCTOR if item['type'].lower()=='conductor' else DIELECTRIC
        native_name=job['staging']['native_material_names'][str(item['id'])]
        mat=doc.AddMaterial(native_name,tuple(item['color_rgb']),kind,item['id'])
        selected.append(native_name);material_records.append((item,mat))
    modeler=VoxelModeler(doc)
    props=StepProperties('VoxelDataImport')
    props.inputFileName=job['staging']['path'];props.newWafer='converted_vdata'
    props.newMaterials=tuple(selected);props.importDopants=False
    modeler.VoxelDataImport(props,OutputParams(doc.GetModel(),'Independent VDATA import',-1))
    wafer=modeler.GetActiveWafers()['converted_vdata']
    actual_resolution=float(wafer.GetResolution())*factor
    if abs(actual_resolution-metadata['voxel_size_um'])>metadata['voxel_size_um']*1e-9:
        raise RuntimeError('Native importer changed the physical voxel spacing')
    native_bounds=list(wafer.GetOctreeBounds())
    info=wafer.GetMaterialBounds();active=[];unused=[]
    for item,mat in material_records:
        volume=float(info.VolumeForMaterial([mat]))*factor**3
        if volume>0:
            mat.SetVisibility(True)
            alias='VDATA_ID_'+str(mat.GetID())
            active.append(dict(source_name=item['name'],source_ids=[item['id']],native_id=mat.GetID(),
                               native_alias=alias,volume_um3=volume,source_color_rgb=item['color_rgb']))
            mat.SetMaterialName(alias)
        else:
            unused.append(item['id'])
    if not active:
        # A valid all-air/void wafer has no surface to export. Record the empty
        # geometry explicitly instead of asking the mesher for zero triangles.
        empty=dict(schema_version=1,status='empty_geometry',input_vdata=metadata['input_vdata'],
                   source_sha256=metadata['source_sha256'],bounds_um=metadata['bounds_um'],
                   translation_um=job['staging']['translation_um'],stl_units='um',materials=[],
                   source_metadata=metadata,independent_input=True,original_project_required=False,
                   process_simulation_run=False,mesh_settings=job['mesh'])
        write_json(output/'conversion_manifest.json',empty)
        report.update(status='empty_geometry',material_count=0,elapsed_seconds=time.monotonic()-start)
        write_json(report_path,report)
        print('EMPTY_GEOMETRY: no positive-volume solid materials',flush=True)
        return
    report.update(status='meshing',native_local_bounds_model_units=native_bounds,
                  unused_palette_ids=unused,active_materials=active)
    write_json(report_path,report)
    mesh=StepProperties('MeshExport');mesh.surfaceMeshOutputFile=str(native/'surface.obj')
    mesh.exportUnits='microns'
    method=job['mesh']['method'];mesh.type='default' if method=='standard' else method
    size=job['mesh']['element_size_um']/factor
    if method=='standard':
        mesh.type.default.globalElementSize=size;mesh.type.default.featureAngle=12
        mesh.type.default.volumeMesh.setEnabled(False);mesh.type.default.advanced.setEnabled(True)
        mesh.type.default.advanced.general.snapToModelBounds=False
        mesh.type.default.advanced.general.markMeshProblemAreas=False
        mesh.type.default.advanced.coarsening.featureScale=job['mesh']['feature_scale']
        mesh.type.default.advanced.remeshing.setEnabled(False)
    elif method=='simple':
        mesh.type.simple.geometryAccuracy=100
    elif method=='delaunay':
        mesh.type.delaunay.globalElementSize=size
        mesh.type.delaunay.volumeMesh.setEnabled(False);mesh.type.delaunay.advanced.setEnabled(True)
        mesh.type.delaunay.advanced.preprocessing.capping=False
    mesh_start=time.monotonic()
    wafer.MeshExport(mesh,OutputParams(doc.GetModel(),'Independent VDATA surface export',-1))
    report['native_meshing_seconds']=time.monotonic()-mesh_start
    vertices,groups=read_obj(native/'surface.obj')
    known={m['native_alias'] for m in active}
    unknown=set(groups)-known-{'Air','Void','Skirt','Unused','Transparent','ReservedMaterial','Feature_Edge'}
    if unknown:
        raise ValueError('Unmapped material groups in native OBJ: '+str(sorted(unknown)))
    meshes=output/'meshes';meshes.mkdir(exist_ok=True)
    translation=job['staging']['translation_um']
    for material in active:
        alias=material['native_alias']
        if alias not in groups or len(groups[alias])==0:
            raise ValueError('Native exporter produced no faces for material '+material['source_name'])
        stl=meshes/('material_{:04d}.stl'.format(material['native_id']))
        audit=audit_and_write(stl,vertices,groups[alias],material['volume_um3'])
        material.update(stl_path=str(stl.resolve()),stl_sha256=sha256(stl),translation_um=translation,audit=audit)
    manifest=dict(schema_version=1,status='geometry_converted',input_vdata=metadata['input_vdata'],
                  source_sha256=metadata['source_sha256'],bounds_um=metadata['bounds_um'],
                  translation_um=translation,stl_units='um',stl_coordinates='local; world = local + translation_um',
                  source_metadata=metadata,materials=active,mesh_settings=job['mesh'],
                  independent_input=True,original_project_required=False,process_simulation_run=False,
                  reserved_palette_entries=[m for m in metadata['materials'] if m['id'] in RESERVED_IDS],
                  limitations=['Native surface export is approximate, not a lossless representation.',
                               'Missing PlaneEquations cannot be uniquely recovered from material fractions.',
                               'Dopant/metrology fields are not converted into optical constitutive models.'])
    write_json(output/'conversion_manifest.json',manifest)
    report.update(status='geometry_converted',elapsed_seconds=time.monotonic()-start,
                  material_count=len(active),solid_triangles=sum(m['audit']['triangles'] for m in active),
                  manifest=str(output/'conversion_manifest.json'),mesh_parameters=str(mesh))
    write_json(report_path,report)
    print('GEOMETRY_CONVERTED {} materials, {} triangles'.format(len(active),report['solid_triangles']),flush=True)


if __name__=='__main__':
    job=read_json(sys.argv[1])
    try:
        convert(job)
    except Exception as exc:
        path=Path(job['output_dir'])/'native_report.json'
        report=read_json(path) if path.exists() else {}
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        write_json(path,report)
        raise
