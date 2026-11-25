"""End-to-end coverage on three artificial ridges, using the public AH53 preset."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import numpy as np
import pytest
from vdata_standalone.storage import read_mesh_archive, write_mesh_archive

spec=spec_from_file_location("ah53_example_straighten",Path(__file__).with_name("straighten.py"))
example=module_from_spec(spec)
spec.loader.exec_module(example)


def synthetic_ridges():
    vertices=[]
    faces=[]
    origin=np.array([16.,-14.,-.5])*1e-6
    yy=[-10.,-8.99,-6.,-3.,0.,3.,6.,8.99,10.]
    zz=[.20,.29,.34,.40,.47,.50]
    for x in (20.,23.2,26.4):
        table={}
        for side in range(2):
            for j,y in enumerate(yy):
                for k,z in enumerate(zz):
                    table[side,j,k]=len(vertices)
                    # A deterministic y-dependent edge offset, plus wall taper.
                    v=[x+side*1.6+.04*np.cos(y)+(.4-z)*.12,y,z]
                    vertices.append(np.asarray(v)*1e-6-origin)
        def quad(keys,reverse=False):
            a,b,c,d=[table[key] for key in keys]
            triangles=[(a,b,c),(a,c,d)]
            faces.extend([t[::-1] for t in triangles] if reverse else triangles)
        for side in range(2):
            for j in range(len(yy)-1):
                for k in range(len(zz)-1):
                    quad([(side,j,k),(side,j+1,k),(side,j+1,k+1),(side,j,k+1)],side==0)
        for j in (0,len(yy)-1):
            for k in range(len(zz)-1):
                quad([(0,j,k),(1,j,k),(1,j,k+1),(0,j,k+1)],j!=0)
        for k in (0,len(zz)-1):
            for j in range(len(yy)-1):
                quad([(0,j,k),(1,j,k),(1,j+1,k),(0,j+1,k)],k==0)
    return np.array(vertices),np.array(faces),origin


def test_two_closed_variants_from_synthetic_vm(tmp_path):
    source=tmp_path/"ridges.vmesh"
    va,fa,t=synthetic_ridges()
    write_mesh_archive(source,va,fa,t,metadata={"material_id":2})
    dest=tmp_path/"straight"
    example.straighten(source,dest)
    counts=[]
    for name in ("B","C"):
        v,f,translation,metadata=read_mesh_archive(dest/(name+".vmesh"))
        report=json.loads((dest/(name+"_geometry.json")).read_text())
        counts.append(len(f))
        assert metadata["variant"]==name and np.array_equal(t,translation)
        assert report["audit"]["watertight_edges"] and report["audit"]["consistent_winding"]
        assert report["audit"]["degenerate_faces"]==0
        assert len(report["walls"])==6
        assert max(w["x_error_bound_B_to_C_nm"] for w in report["walls"])<=1.
        world=(v+t)*1e6
        for edge in report["walls"]:
            selected=(abs(world[:,0]-edge["edge_um"])<.4)&(abs(world[:,1])<=8.5+1.e-10)
            # All vertices placed on each height knot must lie on its fitted line.
            for z,(slope,intercept) in zip(edge["core_z_um"],edge["fit_slope_intercept_um"]):
                pts=world[selected & (abs(world[:,2]-z)<1.e-11)]
                assert len(pts)>=2 and np.max(abs(pts[:,0]-slope*pts[:,1]-intercept))<1.e-10
    assert counts[1]<counts[0]
    with pytest.raises(FileExistsError):
        example.straighten(source,dest)


def test_incompatible_window_leaves_destination_available(tmp_path):
    source=tmp_path/"elsewhere.vmesh"
    v,f,t=synthetic_ridges()
    write_mesh_archive(source,v,f,t+np.array([0.,1.,0.]))
    out=tmp_path/"out"
    with pytest.raises(ValueError,match="AH53 preset"):
        example.straighten(source,out)
    assert not out.exists()
