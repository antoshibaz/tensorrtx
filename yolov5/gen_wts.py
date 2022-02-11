import argparse
import os
import struct

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Convert .pt file to .wts')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-o', '--output', help='Output (.wts) file path (optional)')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid input file')
    if not args.output:
        args.output = os.path.splitext(args.weights)[0] + '.wts'
    elif os.path.isdir(args.output):
        args.output = os.path.join(
            args.output,
            os.path.splitext(os.path.basename(args.weights))[0] + '.wts')
    return args


def convert(pt_file, wts_file):
    # Initialize
    # device = select_device('cpu')
    device = torch.device('cpu')

    # Load model
    model = torch.load(pt_file, map_location=device)['model'].float()  # load to FP32

    # update anchor_grid info
    anchor_grid = model.model[-1].anchors * model.model[-1].stride[..., None, None]
    # model.model[-1].anchor_grid = anchor_grid
    delattr(model.model[-1], 'anchor_grid')  # model.model[-1] is detect layer
    # The parameters are saved in the OrderDict through the "register_buffer" method, and then saved to the weight.
    model.model[-1].register_buffer("anchor_grid", anchor_grid)
    model.to(device).eval()

    with open(wts_file, 'w') as f:
        f.write('{}\n'.format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')

    print('Pytorch model {} conversion to wts file {} is successfully completed'.format(pt_file, wts_file))


def run(args):
    pt_file = args.weights
    wts_file = args.output
    convert(pt_file, wts_file)


if __name__ == '__main__':
    cli_args = parse_args()
    run(cli_args)
