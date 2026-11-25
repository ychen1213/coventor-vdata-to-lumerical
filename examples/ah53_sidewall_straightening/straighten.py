"""Straighten the six long sidewalls of the AH53 example from its indexed VMESH.

Run from the converter repository root with Python's -m option. The AH53
preset uses world coordinates in micrometres. The fitted core extends from
y=-8.5 to +8.5 and from z=0.31 to 0.47. Near its borders four transition
strips connect each new sidewall to the existing surface. B and C share those
borders and differ in the number of triangles used across the central strips.
"""
from pathlib import Path
import argparse
import hashlib
import json
import tempfile
import time
import numpy as np

from vdata_standalone.storage import read_mesh_archive, write_mesh_archive
from vdata_standalone.meshing import audit_mesh
from vdata_standalone.planar_optimize import _patch_components, _one_boundary_loop

EDGES = [20, 21.6, 23.2, 24.8, 26.4, 28]
UPPER = .4889


def snap_grid(values, target):
    candidates = values[np.abs(values-target)<2e-6]
    if not len(candidates):
        raise ValueError(f'AH53 preset requires grid vertices at {target} micrometres')
    unique, counts = np.unique(candidates, return_counts=True)
    return float(unique[counts.argmax()])


def split_upper(local, faces, origin, *, verify=False):
    """Subdivide only triangles straddling the upper z-cut in the six wall boxes."""
    pos = (local+origin)*1e6
    zcut = UPPER
    z = pos[faces,2]
    xv = pos[faces,0].mean(axis=1)
    yv = pos[faces,1].mean(axis=1)
    band = np.logical_or.reduce([np.abs(xv-x)<.4 for x in EDGES])
    y0=snap_grid(pos[:,1],-8.99); y1=snap_grid(pos[:,1],8.99)
    straddled=(z.min(axis=1)<zcut) & (z.max(axis=1)>zcut)
    ids = np.flatnonzero(band & (yv>y0) & (yv<y1) & straddled)
    if not len(ids):
        raise ValueError('No sidewall triangles cross the top of the AH53 window')
    keep = np.ones(len(faces),bool)
    keep[ids]=False
    new_vertices=[]; new_faces=[]; intersections={}; parents=[]
    level=zcut*1e-6-origin[2]
    def hit(i,j):
        a,b=sorted((int(i),int(j)))
        key=(a,b)
        if key not in intersections:
            p,q=local[a],local[b]
            t=(level-p[2])/(q[2]-p[2])
            value=p+(q-p)*t
            value[2]=level
            intersections[key]=len(local)+len(new_vertices)
            new_vertices.append(value)
        return intersections[key]
    for face_id in ids:
        row=faces[face_id]
        above=local[row,2]>level
        k=np.flatnonzero(above if above.sum()==1 else ~above)[0]
        a,b,c=np.roll(row,-int(k))
        ab,ac=hit(a,b),hit(a,c)
        new_faces.extend(((a,ab,ac),(ab,b,c),(ab,c,ac)))
        parents.extend([face_id]*3)
    conformity=0
    for face_id in np.flatnonzero(keep & straddled):
        row=faces[face_id]
        hits=[(k,intersections[tuple(sorted((int(row[k]),int(row[(k+1)%3]))))])
              for k in range(3) if tuple(sorted((int(row[k]),int(row[(k+1)%3])))) in intersections]
        if hits:
            if len(hits)!=1: raise ValueError('Unexpected multiple cuts in neighboring face')
            k,mid=hits[0]; a,b,c=np.roll(row,-k)
            new_faces.extend(((a,mid,c),(mid,b,c))); keep[face_id]=False; conformity+=1
            parents.extend([face_id]*2)
    vout=np.vstack((local,new_vertices)); fout=np.vstack((faces[keep],new_faces))
    basic=vout,fout,int(len(ids)+conformity)
    if not verify: return basic
    parent_ids=np.asarray(parents)
    orig=local[faces[parent_ids]]
    child=vout[np.asarray(new_faces)]
    normal=np.cross(orig[:,1]-orig[:,0],orig[:,2]-orig[:,0])
    norm=np.linalg.norm(normal,axis=1)
    normal/=norm[:,None]
    residual=np.abs(np.einsum('ijk,ik->ij',child-orig[:,:1],normal))
    area2=np.einsum('ij,ij->i',np.cross(child[:,1]-child[:,0],child[:,2]-child[:,0]),normal)
    totals=np.bincount(parent_ids,weights=area2,minlength=len(faces))
    return (*basic, dict(original_triangles_split=len(ids)+conformity,
          single_edge_neighbor_splits=conformity,positive_projected_area=bool(np.all(area2>0)),
          subdivision_max_normal_distance_nm=float(residual.max()*1e9),
          max_relative_area_budget_residual=float(np.max(np.abs(totals[parent_ids]/norm-1)))))


def source_cut(tris,z,yq):
    hits=[]
    for i,j in [(0,1),(1,2),(2,0)]:
        p,q=tris[:,i],tris[:,j]
        live=((p[:,2]<=z)&(q[:,2]>z))|((q[:,2]<=z)&(p[:,2]>z))
        fi=np.flatnonzero(live)
        if len(fi):
            t=((z-p[fi,2])/(q[fi,2]-p[fi,2]))[:,None]
            hits.append((fi,p[fi,:2]+t*(q[fi,:2]-p[fi,:2])))
    if not hits:
        raise ValueError(f'No triangles intersect the required cut at z={z} micrometres')
    fi=np.concatenate([i for i,p in hits])
    ab=np.concatenate([p for i,p in hits])[np.argsort(fi,kind='stable')].reshape(-1,2,2)
    i=np.argmin(ab[:,:,1],axis=1)
    left=ab[np.arange(len(ab)),i]
    right=ab[np.arange(len(ab)),1-i]
    hitcount=np.zeros(len(yq),int)
    result=np.zeros(len(yq)); spreads=[]
    for a,b in zip(left,right):
        lo,hi=np.searchsorted(yq,[a[1],b[1]])
        if lo>=hi: continue
        ys=yq[lo:hi]
        x=a[0]+(b[0]-a[0])*(ys-a[1])/(b[1]-a[1])
        prev=hitcount[lo:hi]>0
        if prev.any(): spreads.append(float(np.max(abs(result[lo:hi][prev]-x[prev]))))
        result[lo:hi]=x; hitcount[lo:hi]+=1
    if np.any(hitcount==0) or max(spreads,default=0)>1e-6:
        raise ValueError(('ambiguous measured edge',z,int((hitcount==0).sum()),max(spreads,default=0)))
    return result


def patch_info(local,faces,origin):
    p=(local+origin)*1e6
    y0=snap_grid(p[:,1],-8.99); y1=snap_grid(p[:,1],8.99)
    z0=snap_grid(p[:,2],.29); z1=UPPER
    centroid=p[faces].mean(axis=1)
    selected=np.zeros(len(faces),bool)
    wall_data=[]
    for edge in EDGES:
        mask=((abs(centroid[:,0]-edge)<.4)&(centroid[:,1]>y0)&(centroid[:,1]<y1)&
              (centroid[:,2]>z0)&(centroid[:,2]<z1))
        sf=faces[mask]; tri=p[sf]
        selected |= mask
        if not len(sf):
            raise ValueError(f'No triangles in the sidewall window at x={edge} micrometres')
        components_result=_patch_components(sf,len(p))
        if components_result is None:
            raise ValueError(f'Nonmanifold patch at x={edge} micrometres')
        components,_,_,ba,bb=components_result
        loop=_one_boundary_loop(ba,bb)
        if loop is None or components!=1:
            raise ValueError(('unexpected patch topology',edge,components,sf.shape))
        rectangle={}
        for axis,value,key,sort_axis in [(1,y0,'left',2),(1,y1,'right',2),(2,z0,'bottom',1),(2,z1,'top',1)]:
            subset=loop[abs(p[loop,axis]-value)<1e-8]
            values=subset[np.argsort(p[subset,sort_axis],kind='stable')]
            rectangle[key]=values
        if sum(map(len,rectangle.values()))!=len(loop)+4: 
            raise ValueError(('boundary is not rectangular',edge,len(loop),{k:len(v) for k,v in rectangle.items()}))
        # Moving only the replacement patch, the boundary indices are reused verbatim.
        yq=np.arange(-8.495,8.50,.01)
        zq=np.linspace(.31,.47,33)
        parameters=[]
        for h in zq:
            x=source_cut(tri,h,yq)
            m,c=np.polyfit(yq,x,1)
            parameters.append((float(m),float(c)))
        area=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])[:,0].sum()
        wall_data.append((edge,rectangle,parameters,np.sign(area),int(mask.sum()),zq))
        print('wall',edge,mask.sum(),'perimeter',len(loop),'source fitted',flush=True)
    return faces[~selected],wall_data,dict(y_min_um=y0,y_max_um=y1,z_min_um=z0,z_max_um=z1)


def zipper(low,high,p,axis):
    """Connect monotonic polylines preserving every boundary vertex."""
    i=j=0
    a=np.array([p[k][axis] for k in low]); b=np.array([p[k][axis] for k in high])
    a=(a-a[0])/(a[-1]-a[0]); b=(b-b[0])/(b[-1]-b[0])
    tris=[]
    while i+1<len(low) or j+1<len(high):
        if j+1==len(high) or (i+1<len(low) and a[i+1] <= b[j+1]):
            tris.append((low[i],low[i+1],high[j])); i+=1
        else:
            tris.append((low[i],high[j+1],high[j])); j+=1
    return np.array(tris,dtype=np.int64)


def assemble(local,origin,unchanged,walls,mode):
    added_local=[]; patches=[]; detail=[]; p=(local+origin)*1e6
    new_global={}
    def vert(x,y,z):
        v=np.array([x,y,z])
        idx=len(local)+len(added_local)
        added_local.append(v*1e-6-origin)
        new_global[idx]=v
        return idx
    class P:
        def __getitem__(self,k): return p[k] if k<len(p) else new_global[k]
    points=P()
    outer_corner_y=(-8.5,8.5)
    for edge,r,params,sign,n_source,zq in walls:
        # Independent triangulations of the fitted height-dependent straight
        # lines. A conservative covariance bound limits their mutual distance.
        max_dm=max(abs(a[0]-b[0]) for a,b in zip(params,params[1:]))
        segments_dense=850
        spacing=min(17., .0036/max(max_dm,1e-18)-17/segments_dense)
        if spacing<=0: raise ValueError('requested 1nm accuracy not attainable with dense reference')
        segments_sparse=max(1,int(np.ceil(17/spacing)))
        rows=[]
        for k,(z,(m,c)) in enumerate(zip(zq,params)):
            # Top/bottom rows match between B and C, leaving collars identical.
            n=segments_dense if mode=='B' or k in (0,len(zq)-1) else segments_sparse
            ys=np.linspace(*outer_corner_y,n+1)
            rows.append([vert(c+m*y,y,z) for y in ys])
        positive=[]
        for lo,hi in zip(rows,rows[1:]): positive.append(zipper(lo,hi,points,1))
        positive.append(zipper(r['bottom'],rows[0],points,1))
        positive.append(zipper(rows[-1],r['top'],points,1))
        positive.append(zipper(r['left'],[row[0] for row in rows],points,2)[:,::-1])
        positive.append(zipper([row[-1] for row in rows],r['right'],points,2)[:,::-1])
        block=np.vstack(positive)
        # Adjacent collars share the seam joining their two corners.
        if sign<0: block=block[:,::-1]
        patches.append(block)
        detail.append(dict(edge_um=edge,source_faces=n_source,new_faces=len(block),
            dense_y_intervals=segments_dense,sparse_y_intervals=segments_sparse,
            core_z_um=zq.tolist(),fit_slope_intercept_um=params,
            x_error_bound_B_to_C_nm=1000*max_dm*(17/segments_dense+17/segments_sparse)/4))
    vnew=np.vstack((local,added_local)); fnew=np.vstack([unchanged,*patches])
    # Original unused patch vertices can now be dropped with a stable remap.
    used, inv=np.unique(fnew.ravel(),return_inverse=True)
    return vnew[used], inv.reshape(-1,3).astype('int32'),detail


def build(source, destination):
    """Write both models into an empty staging directory."""
    start=time.time()
    with source.open('rb') as stream:
        digest=hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    source_sha=digest.hexdigest()
    v,f,t,metadata=read_mesh_archive(source)
    v,f,n,subdivision=split_upper(v,f,t,verify=True)
    if not subdivision['positive_projected_area']:
        raise ValueError('A source subdivision contains an inverted or degenerate face')
    fixed,walls,bounds=patch_info(v,f,t)
    for mode in ('B','C'):
        vout,fout,detail=assemble(v,t,fixed,walls,mode)
        audit=audit_mesh(vout,fout)
        print(mode,'audit',audit,flush=True)
        if not audit['watertight_edges'] or not audit['consistent_winding'] or audit['degenerate_faces']:
            raise ValueError(('invalid candidate',mode,audit))
        if max(w['x_error_bound_B_to_C_nm'] for w in detail)>1.0:
            raise ValueError('Ruled-wall tessellation exceeds the 1 nm lateral error budget')
        result={'source':source.name,'source_sha256':source_sha,
                'bounds':bounds,'core_y_um':[-8.5,8.5],'core_z_um':[.31,.47],
                'split_original_triangles':n,'unchanged_and_subdivided_faces':len(fixed),
                'subdivision':subdivision,'walls':detail,'audit':audit,
                'run_seconds_at_export':time.time()-start}
        (destination/(mode+'_geometry.json')).write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        write_mesh_archive(destination/(mode+'.vmesh'),vout,fout,t,
                           metadata={'material_id':metadata.get('material_id',2),
                                     'variant':mode,'source':source.name,'source_sha256':source_sha})
        print(mode,'ready',flush=True)


def straighten(source, output):
    """Publish a complete B/C archive pair into a new output directory."""
    source=Path(source).resolve(strict=True)
    output=Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Choose a new output directory: {output}')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.straighten-',dir=output.parent) as tmp:
        staged=Path(tmp)/'models'
        staged.mkdir()
        build(source,staged)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f'Output directory already exists: {output}')
        staged.rename(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True,help='Reconstructed AH53 material VMESH')
    parser.add_argument('--output',type=Path,required=True,help='New directory for B/C archives and geometry reports')
    args=parser.parse_args()
    try:
        straighten(args.input,args.output)
    except (ValueError,OSError) as exc:
        parser.exit(1,f'Cannot build AH53 sidewalls: {exc}\n')


if __name__=='__main__':
    main()
