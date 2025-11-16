import argparse

from usdf.utils.watertight_utils import make_mesh_watertight

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a watertight mesh from a .')
    parser.add_argument('input', type=str, help='Input mesh file.')
    parser.add_argument('output', type=str, help='Output mesh file.')
    args = parser.parse_args()

    make_mesh_watertight(args.input, args.output)
