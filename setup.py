import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
from os import path

here = path.abspath(path.dirname(__file__))

# Use correct conda compiler used to build pytorch
os.environ['CXX'] = os.environ.get('GXX', '')

setup(
    name='cosypose',
    version='1.0.0',
    description='CosyPose',
    packages=find_packages(),
    data_files=[("cosypose",
                 ["rclone.conf"]),
                ("cosypose/libmesh/meshlab_templates",
                 ["cosypose/libmesh/meshlab_templates/template_add_uv.mlx",
                  "cosypose/libmesh/meshlab_templates/template_downsample.mlx",
                  "cosypose/libmesh/meshlab_templates/template_downsample_textures.mlx",
                  "cosypose/libmesh/meshlab_templates/template_ply_texture_to_obj.mlx",
                  "cosypose/libmesh/meshlab_templates/template_remesh_marchingcubes.mlx",
                  "cosypose/libmesh/meshlab_templates/template_remesh_poisson.mlx",
                  "cosypose/libmesh/meshlab_templates/template_sample_points.mlx",
                  "cosypose/libmesh/meshlab_templates/template_transfer_texture.mlx",
                  "cosypose/libmesh/meshlab_templates/template_vertexcolor_to_texture.mlx", ])],
    ext_modules=[
        CppExtension(
            name='cosypose_cext',
            sources=[
                'cosypose/csrc/cosypose_cext.cpp'
            ],
            extra_compile_args=['-O3'],
            verbose=True
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
