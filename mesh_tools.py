"""Material-separated OBJ -> compact, audited binary STL. Python 3.6+ NumPy."""
from pathlib import Path
import struct
import numpy as np


def read_obj(path):
    vertices=[];groups={};material=None
    with Path(path).open('r',encoding='utf-8-sig',errors='strict') as stream:
        for line in stream:
            words=line.strip().split()
            if not words or words[0].startswith('#'):
                continue
            if words[0]=='v':
                vertices.append([float(v) for v in words[1:4]])
            elif words[0]=='usemtl':
                material=line.strip().split(None,1)[1]
            elif words[0]=='f':
                if not material:
                    raise ValueError('OBJ face has no material assignment')
                ids=[int(w.split('/')[0]) for w in words[1:]]
                ids=[i-1 if i>0 else len(vertices)+i for i in ids]
                for j in range(1,len(ids)-1):
                    groups.setdefault(material,[]).append([ids[0],ids[j],ids[j+1]])
    v=np.asarray(vertices,dtype=np.float64)
    if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all():
        raise ValueError('OBJ has no finite three-dimensional vertices')
    f={key:np.asarray(value,dtype=np.int64) for key,value in groups.items()}
    for key,a in f.items():
        if a.min()<0 or a.max()>=len(v):
            raise ValueError('Invalid OBJ face index in material '+key)
    return v,f


def audit_and_write(path,vertices,faces,reference_volume_um3):
    # STL uses float32 coordinates. Quantize before the independent topology
    # and volume check so collapsed features cannot silently pass validation.
    triangles=np.asarray(vertices[faces],dtype=np.float32)
    unique,inverse=np.unique(triangles.reshape(-1,3),axis=0,return_inverse=True)
    f=inverse.reshape(-1,3)
    tri=unique[f].astype(np.float64)
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    area2=np.linalg.norm(cross,axis=1)
    bad=int(np.count_nonzero(area2==0))
    directed=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]))
    _,reverse,counts=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True,return_counts=True)
    winding=np.bincount(reverse,weights=np.where(directed[:,0]<directed[:,1],1,-1))
    open_edges=int(np.count_nonzero(counts!=2));bad_winding=int(np.count_nonzero(winding))
    if bad or open_edges or bad_winding:
        raise ValueError('Unsafe surface: degenerate={}, non-closed edges={}, inconsistent edges={}'.format(bad,open_edges,bad_winding))
    local=tri-tri.reshape(-1,3).mean(axis=0)
    volume=float(np.einsum('ij,ij->i',local[:,0],np.cross(local[:,1],local[:,2])).sum()/6.)
    flipped=volume<0
    if flipped:
        triangles=triangles[:,::-1].copy();cross=-cross;volume=-volume
    if not np.isfinite(volume) or volume<=0:
        raise ValueError('Surface does not enclose a positive finite volume')
    records=np.empty(len(triangles),dtype=[('normal','<f4',(3,)),('v','<f4',(3,3)),('attribute','<u2')])
    records['normal']=cross/area2[:,None];records['v']=triangles;records['attribute']=0
    with Path(path).open('wb') as stream:
        stream.write(b'Coventor VDATA converter; local vertex units um; see placement manifest'.ljust(80,b' '))
        stream.write(struct.pack('<I',len(triangles)));stream.write(records.tobytes())
    bounds=[unique.min(axis=0).astype(float).tolist(),unique.max(axis=0).astype(float).tolist()]
    return dict(triangles=len(triangles),vertices=len(unique),closed_edges_all_two=True,
                degenerate_triangles=bad,inconsistent_edges=bad_winding,orientation_flipped=bool(flipped),
                local_bounds_um=bounds,mesh_volume_um3=volume,native_volume_um3=reference_volume_um3,
                volume_difference_percent=100*(volume/reference_volume_um3-1.) if reference_volume_um3 else None,
                stl_bytes=Path(path).stat().st_size,
                limit='Closure and volume checks do not certify every surface point or optical accuracy.')
