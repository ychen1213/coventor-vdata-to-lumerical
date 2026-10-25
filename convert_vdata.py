"""Standalone Coventor VDATA -> per-material STL -> optional verified FDTD.

The original process project is never opened. Python 3.6+ standard library is
sufficient for this entry point; geometry uses the installed Coventor SDK.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import sys
import traceback
from common import (inspect_vdata,stage_standalone,discover_semulator,native_environment,
                    write_json,read_json,run_logged,sha256,make_ascii_work_directory,
                    remove_work_directory,publish_native_geometry)

ROOT=Path(__file__).resolve().parent


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path,help='One standalone Coventor-exported VDATA file')
    p.add_argument('--output',type=Path,help='A new output directory')
    p.add_argument('--inspect',action='store_true',help='Print metadata only; no licensed software is started')
    p.add_argument('--resume',action='store_true',help='Reuse completed geometry for the same input hash/settings')
    p.add_argument('--verify-fdtd',action='store_true',help='Import into a private CAD session and save/reopen imported.fsp')
    p.add_argument('--material-map',type=Path,help='JSON source-name to installed FDTD material mapping')
    p.add_argument('--semulator-root',type=Path)
    p.add_argument('--lumerical-root',type=Path)
    p.add_argument('--api-version',default='9.3',help='Installed SEMulator3D Python API version (default: 9.3)')
    p.add_argument('--mesh-method',choices=('standard','simple','delaunay'),default='standard')
    p.add_argument('--mesh-size-nm',type=float,help='Default: source voxel size; controls native mesher, not physical accuracy')
    p.add_argument('--feature-scale',type=float,default=.2,help='Standard mesh feature scale in voxel units')
    p.add_argument('--keep-global-coordinates',action='store_true',help='Disable local recentering; less suitable for very large origins')
    p.add_argument('--keep-intermediates',action='store_true',help='Keep the native ASCII scratch workspace for diagnostics')
    p.add_argument('--timeout',type=float,default=3600,help='Per native stage timeout seconds (default 3600)')
    args=p.parse_args(argv)
    if args.mesh_size_nm is not None and (not math.isfinite(args.mesh_size_nm) or args.mesh_size_nm<=0):
        p.error('--mesh-size-nm must be positive')
    if not math.isfinite(args.feature_scale) or not math.isfinite(args.timeout) or args.feature_scale<=0 or args.timeout<=0:
        p.error('--feature-scale and --timeout must be positive')
    try:
        metadata=inspect_vdata(args.input)
    except (ValueError,OSError) as exc:
        print('ERROR: '+str(exc),file=sys.stderr);return 1
    if args.inspect:
        print(json.dumps(metadata,indent=2,ensure_ascii=False));return 0
    output=args.output.resolve() if args.output else args.input.resolve().with_name(args.input.stem+'_converted_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if output==args.input.resolve() or (output.exists() and not output.is_dir()):
        p.error('Output must be a directory different from the input')
    mesh=dict(method=args.mesh_method,element_size_um=args.mesh_size_nm/1000. if args.mesh_size_nm else metadata['voxel_size_um'],feature_scale=args.feature_scale)
    settings=dict(mesh=mesh,recenter=not args.keep_global_coordinates,api_version=args.api_version)
    nonempty=output.exists() and any(output.iterdir())
    if nonempty and not args.resume:
        p.error('Output is not empty. Choose a new directory or use --resume for the same input/settings.')
    output.mkdir(parents=True,exist_ok=True)
    summary=dict(schema_version=1,status='starting',input_vdata=metadata['input_vdata'],
                 source_sha256=metadata['source_sha256'],output_dir=str(output),independent_input=True,
                 original_project_required=False,settings=settings)
    summary_path=output/'conversion_report.json'
    existing=read_json(summary_path) if args.resume and summary_path.exists() else None
    if nonempty and args.resume:
        if not existing or existing.get('source_sha256')!=metadata['source_sha256'] or existing.get('settings')!=settings:
            p.error('--resume requires the same source hash and native conversion settings')
    write_json(summary_path,summary)
    work=None;work_parent=None
    try:
        manifest_path=output/'conversion_manifest.json'
        if args.resume and manifest_path.exists():
            manifest=read_json(manifest_path)
            if manifest['source_sha256']!=metadata['source_sha256'] or not all(Path(m['stl_path']).is_file() for m in manifest['materials']):
                raise ValueError('Existing geometry is incomplete or belongs to another input')
            if any(m.get('stl_sha256') and sha256(m['stl_path'])!=m['stl_sha256'] for m in manifest['materials']):
                raise ValueError('An exported STL was changed after native conversion; refusing to reuse it')
            print('Reusing verified native geometry for the identical input.',flush=True)
        else:
            if args.resume:
                raise ValueError('Native geometry is incomplete; choose a fresh output directory to retry safely')
            work,work_parent=make_ascii_work_directory()
            staging=stage_standalone(args.input,work/'standalone_input.vdata',metadata,not args.keep_global_coordinates)
            shutil.copy2(staging['path'],str(output/'standalone_input.vdata'))
            job=dict(output_dir=str(work),metadata=metadata,staging=staging,mesh=mesh)
            write_json(output/'native_job.json',job)
            write_json(work/'native_job.json',job)
            install,python=discover_semulator(args.semulator_root)
            env=native_environment(install,python,work,args.api_version)
            print('Importing standalone VDATA and extracting surfaces; no source project is loaded.',flush=True)
            process=run_logged([python,ROOT/'native_worker.py',work/'native_job.json'],output/'native.log',env,work,args.timeout)
            summary['native_process']=process
            publish_native_geometry(work,output)
            if process['returncode'] or not manifest_path.exists():
                native_report=read_json(output/'native_report.json') if (output/'native_report.json').exists() else {}
                raise RuntimeError(native_report.get('error','Native SDK failed (exit {}); see native.log'.format(process['returncode'])))
            manifest=read_json(manifest_path)
        if not manifest['materials']:
            summary.update(status='empty_geometry',geometry_manifest=str(manifest_path),
                           material_count=0,source_file_unchanged=sha256(args.input)==metadata['source_sha256'])
            write_json(summary_path,summary)
            print('Valid empty geometry: the file contains no positive-volume solid material. No STL or FDTD object is needed.',flush=True)
            return 0
        from fdtd_bridge import build
        fdtd=build(manifest_path,output/'fdtd',material_map=args.material_map,
                   lumerical_root=args.lumerical_root,verify=args.verify_fdtd)
        summary['fdtd_report']=fdtd
        summary['source_file_unchanged']=sha256(args.input)==metadata['source_sha256']
        if not summary['source_file_unchanged']:
            raise RuntimeError('Source hash changed during conversion')
        # Geometry conversion and optical material mapping are distinct stages.
        status=fdtd.get('fdtd_status')
        summary.update(status='complete' if status in ('verified','ready','ready_to_import','generated') else 'geometry_converted',
                       geometry_manifest=str(manifest_path),material_count=len(manifest['materials']))
        write_json(summary_path,summary)
        print('Geometry: {} materials -> {}'.format(len(manifest['materials']),output/'meshes'),flush=True)
        print('FDTD status: {}. Report: {}'.format(status,summary_path),flush=True)
        if args.verify_fdtd and status!='verified':
            return 3
        return 0
    except Exception as exc:
        summary.update(status='failed',error=str(exc),traceback=traceback.format_exc(),
                       source_file_unchanged=sha256(args.input)==metadata['source_sha256'])
        write_json(summary_path,summary)
        print('ERROR: {}\nSee {}'.format(exc,summary_path),file=sys.stderr,flush=True)
        return 1
    finally:
        if work is not None:
            if args.keep_intermediates:
                print('Native intermediates retained: '+str(work),flush=True)
            else:
                try:
                    remove_work_directory(work,work_parent)
                except OSError as cleanup_error:
                    print('Native temporary directory retained ({}): {}'.format(cleanup_error,work),flush=True)


if __name__=='__main__':
    sys.exit(main())
