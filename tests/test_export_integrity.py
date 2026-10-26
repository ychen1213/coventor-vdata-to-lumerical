"""Export integrity: no accidental overwrite, batch preflight and rollback."""
from pathlib import Path

import numpy as np
import pytest

from test_cli import write_fixture
from vdata_standalone import cli, meshing, planar


def converted(tmp_path, *, multi=False):
    source=tmp_path/'synthetic.vdata'
    write_fixture(source,with_second_material=multi)
    out=tmp_path/'converted'
    assert cli.main(['convert',str(source),'--output',str(out),'--storage','compact'])==0
    return out


def snapshot(out):
    return {str(p.relative_to(out)):p.read_bytes() for p in out.rglob('*') if p.is_file()}


def triangle():
    return (np.array([[0.,0.,0.],[1e-9,0.,0.],[0.,1e-9,0.]]),
            np.array([[0,1,2]],dtype=np.int32),np.zeros(3))


@pytest.mark.parametrize('command', ['export-stl','lsf'])
def test_modified_existing_stl_refused_without_adopting_its_hash(tmp_path,command):
    out=converted(tmp_path)
    assert cli.main(['export-stl',str(out)])==0
    stl=out/'meshes/material_0002.stl'
    data=bytearray(stl.read_bytes());data[-10]^=1;stl.write_bytes(data)
    before=snapshot(out)
    assert cli.main([command,str(out)])==2
    assert snapshot(out)==before


@pytest.mark.parametrize('command', ['export-stl','export-planar'])
def test_corrupt_second_material_preflight_leaves_no_partial_batch(tmp_path,command):
    out=converted(tmp_path,multi=True)
    with (out/'meshes/material_0003.vmesh').open('ab') as stream:
        stream.write(b'corruption')
    before=snapshot(out)
    assert cli.main([command,str(out)])==2
    assert snapshot(out)==before


@pytest.mark.parametrize('command,suffix', [('export-stl','.stl'),('export-planar','_planar.mat')])
def test_unrecorded_second_target_preserved_before_any_export(tmp_path,command,suffix):
    out=converted(tmp_path,multi=True)
    target=out/('meshes/material_0003'+suffix)
    target.write_bytes(b'User data, do not overwrite')
    before=snapshot(out)
    assert cli.main([command,str(out)])==2
    assert snapshot(out)==before


@pytest.mark.parametrize('command,module,name', [
    ('export-stl',meshing,'write_binary_stl'),
    ('export-planar',planar,'write_planar_mat')])
def test_failure_mid_batch_rolls_back_only_new_files_and_can_retry(tmp_path,monkeypatch,command,module,name):
    out=converted(tmp_path,multi=True)
    before=snapshot(out)
    original=getattr(module,name)
    calls=[]
    def fail_second(*args,**kwargs):
        calls.append(str(args[0]))
        if len(calls)==2:
            raise OSError('Simulated second material export failure')
        return original(*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(module,name,fail_second)
        assert cli.main([command,str(out)])==2
    assert len(calls)==2
    assert snapshot(out)==before
    assert cli.main([command,str(out)])==0


@pytest.mark.parametrize('command,module,name,script_name', [
    ('export-stl',cli,'generate_lsf','import_lumerical.lsf'),
    ('export-planar',planar,'generate_planar_lsf','import_lumerical_planar.lsf')])
def test_script_write_failure_restores_text_and_rolls_back_geometry(tmp_path,monkeypatch,command,module,name,script_name):
    out=converted(tmp_path)
    before=snapshot(out)
    def fail(*args,**kwargs):
        (out/script_name).write_text('partial script',encoding='utf-8')
        raise OSError('Simulated script write failure')
    with monkeypatch.context() as patch:
        patch.setattr(module,name,fail)
        assert cli.main([command,str(out)])==2
    assert snapshot(out)==before
    assert cli.main([command,str(out)])==0


@pytest.mark.parametrize('kind', ['stl','mat'])
def test_writer_never_replaces_existing_destination(tmp_path,kind):
    target=tmp_path/('existing.'+kind)
    target.write_bytes(b'preserved')
    v,f,t=triangle()
    with pytest.raises(FileExistsError):
        if kind=='stl': meshing.write_binary_stl(target,v,f)
        else: planar.write_planar_mat(target,v,f,t)
    assert target.read_bytes()==b'preserved'
    assert list(tmp_path.iterdir())==[target]


@pytest.mark.parametrize('kind', ['stl','mat'])
def test_atomic_writer_preserves_destination_created_concurrently(tmp_path,monkeypatch,kind):
    target=tmp_path/('raced.'+kind)
    v,f,t=triangle()
    module=meshing if kind=='stl' else planar
    original=module.os.link
    def raced_link(source,destination,*args,**kwargs):
        Path(destination).write_bytes(b'created by another writer')
        return original(source,destination,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(module.os,'link',raced_link)
        with pytest.raises(FileExistsError):
            if kind=='stl': meshing.write_binary_stl(target,v,f)
            else: planar.write_planar_mat(target,v,f,t)
    assert target.read_bytes()==b'created by another writer'
    assert list(tmp_path.iterdir())==[target]
