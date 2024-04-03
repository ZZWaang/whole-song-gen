from argparse import ArgumentParser
from params import PARAMS_DICTS
import os
from train.train_config import LdmTrainConfig


def init_parser():
    parser = ArgumentParser(description='train (or resume training) a diffusion model')
    parser.add_argument(
        "--output_dir",
        default='results',
        help='directory in which to store model checkpoints and training logs'
    )
    parser.add_argument("--mode", help="which model to train (frm, ctp, lsh, acc)")
    parser.add_argument('--external', action='store_true', help="whether to use external control")
    parser.add_argument('--autoreg', action='store_true', help="whether to use autoreg control")
    parser.add_argument('--mask_bg', action='store_true', help="whether to mask background-cond at random")
    parser.add_argument('--multi_label', action='store_true', help="whether to use all human phrase labels")
    parser.add_argument('--uniform_pitch_shift', action='store_true',
                        help="whether to apply pitch shift uniformly (as opposed to randomly)")
    parser.add_argument('--debug', action='store_true', help="whether to use a toy dataset")

    return parser


def args_check(args):
    assert args.mode in ['frm', 'ctp', 'lsh', 'acc']
    if args.mode == 'frm':
        assert not args.autoreg and not args.external and not args.mask_bg


def args_setting_to_fn(args):
    def to_str(x: bool, char):
        return char if x else ''

    mode = args.mode
    autoreg = to_str(args.autoreg, 'a')
    external = to_str(args.external, 'e')
    mask_bg = to_str(args.mask_bg, 'b')
    multi_label = to_str(args.multi_label, 'l')
    p_shift = to_str(args.uniform_pitch_shift, 'p')
    debug = to_str(args.debug, 'd')

    return f"{mode}-{autoreg}{external}-{mask_bg}{multi_label}{p_shift}-{debug}"


if __name__ == "__main__":

    parser = init_parser()
    args = parser.parse_args()
    args_check(args)

    random_pitch_aug = not args.uniform_pitch_shift

    params = PARAMS_DICTS[args.mode]
    if args.debug:
        params.override({'batch_size': 2})

    fn = args_setting_to_fn(args)

    output_dir = os.path.join(args.output_dir, fn)
    config = LdmTrainConfig(params, output_dir, args.mode, args.autoreg, args.external,
                            args.mask_bg, args.multi_label, random_pitch_aug, args.debug)

    config.train()
