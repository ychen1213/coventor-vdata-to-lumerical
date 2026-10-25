"""Standalone VDATA conversion followed by optional, audited single-material QEM.

Full-resolution conversion and simplification have separate output directories.
The original input and full-resolution mesh are retained unchanged.
"""
import argparse
from datetime import datetime
import importlib.util
import math
from pathlib import Path
import sys
import traceback

from common import inspect_vdata,read_json,sha256,write_json
import convert_vdata


def check_qem_dependencies():
    if sys.version_info < (3,10):
        raise ValueError('QEM requires Python 3.10+; use a newer interpreter. Full conversion remains available on Python 3.6+.')
    missing=[name for name in ['numpy','scipy','fast_simplification','trimesh','rtree'] if importlib.util.find_spec(name) is None]
    if missing:
        raise ValueError('Missing QEM dependencies: {}. Install requirements-qem.txt with this Python interpreter.'.format(', '.join(missing)))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--reduction',type=float,default=.45,help='Fraction of triangles to remove, e.g. 0.45 or 0.65; zero uses the full converter')
    parser.add_argument('--sampled-max-error-nm',type=float,help='Optional sampled-distance rejection threshold, NOT a continuous-surface error guarantee')
    parser.add_argument('--verify-fdtd',action='store_true')
    parser.add_argument('--material-map',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--inspect',action='store_true')
    parser.add_argument('--semulator-root',type=Path)
    parser.add_argument('--lumerical-root',type=Path)
    parser.add_argument('--api-version',default='9.3')
    parser.add_argument('--mesh-method',choices=('standard','simple','delaunay'),default='standard')
    parser.add_argument('--mesh-size-nm',type=float)
    parser.add_argument('--feature-scale',type=float,default=.2)
    parser.add_argument('--keep-global-coordinates',action='store_true')
    parser.add_argument('--keep-intermediates',action='store_true')
    parser.add_argument('--timeout',type=float,default=3600)
    args=parser.parse_args(argv)
    if not math.isfinite(args.reduction) or not 0<=args.reduction<1:
        parser.error('--reduction must be finite and in [0, 1)')
    if args.sampled_max_error_nm is not None and (not math.isfinite(args.sampled_max_error_nm) or args.sampled_max_error_nm<=0):
        parser.error('--sampled-max-error-nm must be positive')
    if args.inspect:return convert_vdata.main([str(args.input),'--inspect'])
    output=args.output.resolve() if args.output else args.input.resolve().with_name(args.input.stem+'_qem_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    native=output if args.reduction==0 else output/'full_resolution'
    command=[str(args.input),'--output',str(native),'--api-version',args.api_version,
             '--mesh-method',args.mesh_method,'--feature-scale',str(args.feature_scale),'--timeout',str(args.timeout)]
    for flag,value in [('--mesh-size-nm',args.mesh_size_nm),('--semulator-root',args.semulator_root),
                       ('--lumerical-root',args.lumerical_root),('--material-map',args.material_map)]:
        if value is not None:command.extend([flag,str(value)])
    for flag,enabled in [('--resume',args.resume),('--keep-global-coordinates',args.keep_global_coordinates),
                         ('--keep-intermediates',args.keep_intermediates),('--verify-fdtd',args.verify_fdtd and args.reduction==0)]:
        if enabled:command.append(flag)
    if args.reduction==0:return convert_vdata.main(command)
    try:
        metadata=inspect_vdata(args.input)
        check_qem_dependencies()  # Fail before a potentially expensive SDK stage.
        if output.exists() and (not output.is_dir() or (any(output.iterdir()) and not args.resume)):
            raise ValueError('Output is not empty. Choose a new directory or resume the identical input/settings.')
        if args.resume and output.exists() and any(output.iterdir()) and not (native/'conversion_report.json').is_file():
            raise ValueError('This directory has no matching full-resolution conversion to resume')
        previous_path=output/'conversion_report.json'
        if args.resume and previous_path.is_file():
            previous=read_json(previous_path)
            if previous.get('source_sha256')!=metadata['source_sha256'] or previous.get('reduction')!=args.reduction or previous.get('sampled_max_error_nm')!=args.sampled_max_error_nm:
                raise ValueError('Resume requires the identical VDATA, reduction and sampled threshold. Choose a new output for different settings.')
        output.mkdir(parents=True,exist_ok=True)
    except (ValueError,OSError) as exc:
        print('ERROR: '+str(exc),file=sys.stderr);return 1
    report=dict(schema_version=1,status='starting',mode='qem',reduction=args.reduction,sampled_max_error_nm=args.sampled_max_error_nm,
                input_vdata=metadata['input_vdata'],source_sha256=metadata['source_sha256'],
                full_resolution_directory=str(native),result_directory=str(output/'reduced'),
                optical_solver_run=False)
    report_path=output/'conversion_report.json'
    write_json(report_path,report)
    try:
        print('Stage 1/2: standalone VDATA to full-resolution material meshes.',flush=True)
        code=convert_vdata.main(command)
        if code:raise RuntimeError('Full-resolution stage failed; see full_resolution/conversion_report.json')
        full_report=read_json(native/'conversion_report.json')
        if full_report.get('status')=='empty_geometry':
            report.update(status='empty_geometry',material_count=0,source_file_unchanged=sha256(args.input)==metadata['source_sha256'])
            write_json(report_path,report);return 0
        from reduce_conversion import main as reduce_main
        command=[str(native/'conversion_manifest.json'),'--output',str(output/'reduced'),'--reduction',str(args.reduction)]
        for flag,value in [('--sampled-max-error-nm',args.sampled_max_error_nm),('--material-map',args.material_map),
                           ('--lumerical-root',args.lumerical_root)]:
            if value is not None:command.extend([flag,str(value)])
        if args.verify_fdtd:command.append('--verify-fdtd')
        if args.resume and (output/'reduced/reduction_report.json').is_file():command.append('--resume')
        print('Stage 2/2: audited QEM; sampled distances are not a global error guarantee.',flush=True)
        code=reduce_main(command)
        fdtd_path=output/'reduced/fdtd/fdtd_report.json'
        if not fdtd_path.is_file():raise RuntimeError('QEM stage failed; see reduced/reduction_report.json')
        fdtd=read_json(fdtd_path)
        reduction_report=read_json(output/'reduced/reduction_report.json')
        if code not in (0,3) or reduction_report.get('status')=='failed':
            raise RuntimeError(reduction_report.get('error','QEM stage did not finish successfully'))
        if sha256(args.input)!=metadata['source_sha256']:raise RuntimeError('Input VDATA changed during conversion')
        status=fdtd.get('fdtd_status')
        report.update(status='complete' if status in ('verified','ready','ready_to_import','generated') else 'geometry_converted',
                      fdtd_report=fdtd,geometry_manifest=str(output/'reduced/conversion_manifest.json'),
                      reduction_report=str(output/'reduced/reduction_report.json'),source_file_unchanged=True)
        write_json(report_path,report)
        print('QEM output: '+str(output/'reduced'),flush=True)
        return code
    except Exception as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc(),
                      source_file_unchanged=sha256(args.input)==metadata['source_sha256'])
        write_json(report_path,report)
        print('ERROR: {}\nSee {}'.format(exc,report_path),file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':sys.exit(main())
