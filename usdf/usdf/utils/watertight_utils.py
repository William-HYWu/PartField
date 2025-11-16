import argparse
import os
import subprocess

MANIFOLD_EXECUTABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../3rd/Manifold/build/manifold")


def make_mesh_watertight(input_mesh, output_mesh):
    """
    Make a mesh watertight using the `manifold` tool.

    Args:
        input_mesh: file name of the input mesh
        output_mesh: file name of the output mesh
    """
    manifold_cmd = '{} {} {}'.format(MANIFOLD_EXECUTABLE, input_mesh, output_mesh)
    subprocess.call(manifold_cmd, shell=True)
