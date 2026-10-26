"""Independent lossless storage invariants, malformed inputs and streaming."""
import hashlib
import json
import zipfile

import numpy as np
import pytest

from vdata_standalone.storage import (StorageError, write_mesh_archive,
    read_mesh_archive, inspect_mesh_archive, iter_mesh_array_blocks)


def sample():
    v = np.array([[0., -0., 1e-9], [1e-10, -2e-6, 3e-8],
                  [np.nextafter(1., 2.), np.finfo(float).tiny, 1e-300],
                  [2., 3., 4.]], dtype=np.float64)
    f = np.array([[2, 1, 0], [3, 0, 1]], dtype=np.int32)
    t = np.array([1e6, -0., -1e6], dtype=np.float64)
    return v, f, t


@pytest.mark.parametrize('codec', ['lzma', 'zlib'])
@pytest.mark.parametrize('faces_dtype', ['<i4', '<i8', '>u4', '>i8', 'u1', 'u2'])
def test_bit_exact_roundtrip_and_per_array_stream(tmp_path, codec, faces_dtype):
    vertices, faces, translation = sample()
    faces = faces.astype(faces_dtype)
    destination = tmp_path / '中文.vmesh'
    result = write_mesh_archive(destination, vertices, faces, translation,
                                {'material_id': 2, 'name': '硅'}, codec=codec, chunk_rows=2)
    decoded = read_mesh_archive(destination)
    for name, before, after in zip(('vertices_local_m','faces','translation_m'),
                                  (vertices,faces,translation),decoded[:3]):
        assert before.dtype == after.dtype
        assert before.shape == after.shape
        assert before.tobytes() == after.tobytes()
        blocks = list(iter_mesh_array_blocks(destination,name))
        assert b''.join(block.tobytes() for block in blocks) == before.tobytes()
        assert result['array_sha256'][name] == hashlib.sha256(before.tobytes()).hexdigest()
    assert decoded[3] == {'material_id': 2, 'name': '硅'}
    assert result['lossless'] and not result['geometry_changed']
    assert inspect_mesh_archive(destination)['arrays']['faces']['dtype'] == faces.dtype.str


def test_big_endian_float_and_noncontiguous_input(tmp_path):
    v, f, t = sample()
    v = v.astype('>f8')[::-1]
    t = t.astype('>f8')
    path = tmp_path/'big.vmesh'
    write_mesh_archive(path,v,f,t,chunk_rows=1)
    a,b,c,_=read_mesh_archive(path)
    assert a.dtype == v.dtype and a.tobytes() == v.tobytes()
    assert b.tobytes() == f.tobytes() and c.tobytes() == t.tobytes()


def test_random_finite_float_bits_and_modular_wrap(tmp_path):
    rng=np.random.default_rng(11)
    bits=rng.integers(0,2**64-1,size=(1200,3),dtype=np.uint64)
    bits &= np.uint64(0xFFEFFFFFFFFFFFFF)  # finite exponent, retain signs/fractions
    v=bits.view(np.float64)
    f=rng.integers(0,len(v),size=(5000,3),dtype=np.int64)
    t=np.array([0.,-0.,np.nextafter(0.,1.)])
    path=tmp_path/'random.vmesh'
    write_mesh_archive(path,v,f,t,chunk_rows=73)
    result=read_mesh_archive(path)
    for before,after in zip((v,f,t),result):
        assert before.tobytes() == after.tobytes()


def test_empty_mesh_and_no_overwrite(tmp_path):
    p=tmp_path/'empty.vmesh'
    write_mesh_archive(p,np.empty((0,3)),np.empty((0,3),dtype=np.int32),np.zeros(3))
    v,f,t,_=read_mesh_archive(p)
    assert v.shape == f.shape == (0,3)
    assert list(iter_mesh_array_blocks(p,'faces')) == []
    original=p.read_bytes()
    with pytest.raises(FileExistsError):
        write_mesh_archive(p,*sample())
    assert p.read_bytes() == original
    assert not list(tmp_path.glob('.vmesh-*.tmp'))


def alter_archive(path, modifier):
    with zipfile.ZipFile(path) as z:
        contents={name:z.read(name) for name in z.namelist()}
    manifest=json.loads(contents['manifest.json'])
    modifier(contents,manifest)
    contents['manifest.json']=json.dumps(manifest).encode()
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        for name,data in contents.items(): z.writestr(name,data)


def test_block_checksum_and_whole_array_checksum_checked(tmp_path):
    p=tmp_path/'bad.vmesh'
    write_mesh_archive(p,*sample())
    alter_archive(p,lambda c,m:m['arrays']['faces'].__setitem__('sha256','0'*64))
    with pytest.raises(StorageError,match='array SHA256'):
        read_mesh_archive(p)
    alter_archive(p,lambda c,m:m['arrays']['faces']['blocks'][0].__setitem__('sha256','0'*64))
    with pytest.raises(StorageError,match='block SHA256'):
        read_mesh_archive(p)


def test_memory_cap_and_unsupported_format(tmp_path):
    p=tmp_path/'bad.vmesh'
    write_mesh_archive(p,*sample())
    with pytest.raises(StorageError,match='size limit'):
        read_mesh_archive(p,max_raw_bytes=10)
    alter_archive(p,lambda c,m:m.__setitem__('version',999))
    with pytest.raises(StorageError,match='Unsupported'):
        read_mesh_archive(p)


def test_trailing_compressed_stream_rejected(tmp_path):
    p=tmp_path/'bad.vmesh'
    write_mesh_archive(p,*sample())
    def append(c,m):
        block=m['arrays']['faces']['blocks'][0]
        c[block['member']] += b'extraneous'
        block['compressed_bytes'] += 10
    alter_archive(p,append)
    with pytest.raises(StorageError,match='trailing'):
        read_mesh_archive(p)


@pytest.mark.parametrize('change', ['float32','bad_index','nan','bad_translation','bad_codec'])
def test_invalid_geometry_or_options_rejected(tmp_path,change):
    v,f,t=sample()
    kwargs={}
    if change=='float32': v=v.astype(np.float32)
    if change=='bad_index': f[0,0]=99
    if change=='nan': v[0,0]=np.nan
    if change=='bad_translation': t=t[:2]
    if change=='bad_codec': kwargs['codec']='unknown'
    with pytest.raises(StorageError): write_mesh_archive(tmp_path/'a.vmesh',v,f,t,**kwargs)
