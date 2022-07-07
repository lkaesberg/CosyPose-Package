from pathlib import Path
import shutil
from PIL import Image
import PIL
import os
import numpy as np
from plyfile import PlyData
import pymeshlab


def _get_template(template_name):
    template_path = Path(__file__).resolve().parent / 'meshlab_templates' / template_name
    return template_path.read_text()


def run_meshlab_script(in_path, out_path, script, cd_dir=None, has_textures=True):
    in_path = Path(in_path)
    out_path = Path(out_path)
    print(in_path.as_posix())
    print(out_path.as_posix())
    n = np.random.randint(1e6)
    script_path = Path(f'/dev/shm/{n}.mlx')
    script_path.write_text(script)
    # TODO works when removing all parameters
    if cd_dir is None:
        cd_dir = '.'
    command = [f'cd {cd_dir} &&', 'LC_ALL=C',
               'meshlabserver', '-i', in_path.as_posix(), '-o', out_path.as_posix(), '-m', 'vn']
    if has_textures:
        command += ['wt', 'vt']
    command += ['-s', script_path.as_posix()]
    print(' '.join(command))
    os.system(' '.join(command))
    script_path.unlink()
    return


def convert_to_obj(in_path, out_path, has_textures=True, out_texture_path=None):
    in_path = Path(in_path)
    out_path = Path(out_path)
    ms = pymeshlab.MeshSet()
    if in_path.suffix.lower() in [".gltf", ".glb"]:
        ms.load_new_mesh(in_path.as_posix(), load_in_a_single_layer=True)
    else:
        ms.load_new_mesh(in_path.as_posix())

    if has_textures:
        ms.convert_pervertex_uv_into_perwedge_uv()
    else:
        ms.parametrization_trivial_per_triangle(border=0, method=0)
        ms.transfer_vertex_color_to_texture(textname=out_texture_path)

    ms.save_current_mesh(out_path.as_posix())
    return


def add_texture_to_mtl(obj_path):
    # Sometimes meshlab forgets to puts the texture in the output mtl.
    obj_path = Path(obj_path)
    obj_path.with_suffix('.obj.mtl').touch(exist_ok=True)
    texture_name = obj_path.with_suffix('').name + '_texture.png'
    mtl_path = obj_path.with_suffix('.obj.mtl')
    mtl = mtl_path.read_text()
    mtl += f'\nmap_Kd {texture_name}'
    mtl_path.write_text(mtl)
    return


def ply_to_obj(ply_path, obj_path, texture_size=(1024, 1024)):
    ply_path = Path(ply_path)
    obj_path = Path(obj_path)
    ply_copied_path = obj_path.parent / ply_path.name
    is_same = ply_copied_path == ply_path
    print(ply_path)
    print(ply_copied_path)
    print(obj_path)
    if not is_same:
        print("Copy")
        shutil.copy(ply_path, ply_copied_path)

    ply = PlyData.read(ply_path)
    ply_texture = None
    for c in ply.comments:
        if 'TextureFile' in c:
            ply_texture = c.split(' ')[-1]

    if ply_texture is None:
        template = _get_template('template_vertexcolor_to_texture.mlx')
        out_texture_path = obj_path.with_suffix('').name + '_texture.png'
        script = template.format(out_texture_path=out_texture_path)
        # run_meshlab_script(ply_copied_path, obj_path, script, cd_dir=obj_path.parent)
        convert_to_obj(ply_path, obj_path, False, out_texture_path)
    else:
        template = _get_template('template_ply_texture_to_obj.mlx')
        script = template
        ply_texture_name = ply_texture.split('.')[0]
        out_texture_path = obj_path.parent / (ply_texture_name + '_texture.png')
        out_texture_path.parent.mkdir(exist_ok=True)
        shutil.copy(ply_path.parent / ply_texture, out_texture_path)
        Image.open(out_texture_path).resize(texture_size, resample=PIL.Image.BILINEAR).save(out_texture_path)
        # run_meshlab_script(ply_path, obj_path, template)
        convert_to_obj(ply_path, obj_path)
        add_texture_to_mtl(obj_path)
    if not is_same:
        ply_copied_path.unlink()
    return


def downsample_obj(in_path, out_path, n_faces=1000):
    # Remesh and downsample
    template = _get_template('template_downsample_textures.mlx')
    script = template.format(n_faces=n_faces)
    run_meshlab_script(in_path, out_path, script, has_textures=True, cd_dir=in_path.parent)


def sample_points(in_path, out_path, n_points=2000):
    # Remesh and downsample
    template_path = Path(__file__).resolve().parent / 'template_sample_points.mlx'
    template = template_path.read_text()
    script = template.format(n_points=n_points)
    run_meshlab_script(in_path, out_path, script)
    return
