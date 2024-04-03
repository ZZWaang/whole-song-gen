from experiments.whole_song_gen import WholeSongGeneration
import torch
from argparse import ArgumentParser


DEFAULT_FRM_MODEL_FOLDER = 'results_default/frm---/v-default'
DEFAULT_CTP_MODEL_FOLDER = 'results_default/ctp-a-b-/v-default'
DEFAULT_LSH_MODEL_FOLDER = 'results_default/lsh-a-b-/v-default'
DEFAULT_ACC_MODEL_FOLDER = 'results_default/acc-a-b-/v-default'

DEFAULT_DEMO_DIR = 'demo'


def init_parser():
    parser = ArgumentParser(description='inference a whole-song generation experiment')
    parser.add_argument(
        "--demo_dir",
        default=DEFAULT_DEMO_DIR,
        help='directory in which to generated samples'
    )
    parser.add_argument("--mpath0", default=DEFAULT_FRM_MODEL_FOLDER, help="Form generation model path")
    parser.add_argument("--mid0", default='default', help="Form generation model id")

    parser.add_argument("--mpath1", default=DEFAULT_CTP_MODEL_FOLDER, help="Counterpoint generation model path")
    parser.add_argument("--mid1", default='default', help="Counterpoint generation model id")

    parser.add_argument("--mpath2", default=DEFAULT_LSH_MODEL_FOLDER, help="Lead Sheet generation model path")
    parser.add_argument("--mid2", default='default', help="Lead Sheet generation model id")

    parser.add_argument("--mpath3", default=DEFAULT_ACC_MODEL_FOLDER, help="Accompaniment generation model path")
    parser.add_argument("--mid3", default='default', help="Accompaniment generation model id")

    parser.add_argument("--nsample", default=1, type=int, help="Number of generated samples")

    parser.add_argument("--pstring", help="Specify phrase structure. If specified, key must be specified.")

    parser.add_argument("--nbpm", default=4, type=int, help="Number of beats per measure")

    parser.add_argument("--key", default=0, type=int, help="Tonic of the key (0 - 11)")

    parser.add_argument('--minor', action='store_false', help="Whether to generated in minor key.")

    parser.add_argument('--debug', action='store_true', help="Whether to use a toy dataset")

    return parser


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    whole_song_expr = WholeSongGeneration.init_pipeline(
        frm_model_folder=args.mpath0,
        ctp_model_folder=args.mpath1,
        lsh_model_folder=args.mpath2,
        acc_model_folder=args.mpath3,
        frm_model_id=args.mid0,
        ctp_model_id=args.mid1,
        lsh_model_id=args.mid2,
        acc_model_id=args.mid3,
        debug_mode=args.debug,
        device=None
    )

    whole_song_expr.main(
        n_sample=args.nsample,
        nbpm=4,
        nspb=4,
        phrase_string=args.pstring,
        key=args.key,
        is_major=args.minor,
        demo_dir=args.demo_dir
    )
