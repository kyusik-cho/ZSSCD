# Common evaluation arguments
from argparse import ArgumentParser
import torch
from deva.model.network import DEVA


def add_deva_args(parser):
    parser.add_argument('--devamodel', default='./model_weights/DEVA-propagation.pth')    
    
    parser.add_argument('--max_missed_detection_count', default=5)
    # parser.add_argument('--no_metrics', default=False)
    parser.add_argument('--max_num_objects', default=200)
    parser.add_argument('--postprocess_limit_max_id', default=20)
    

    parser.add_argument('--output', default=None)
    parser.add_argument(
        '--save_all',
        action='store_true',
        help='Save all frames',
    )

    parser.add_argument('--amp', action='store_true')

    # Model parameters
    parser.add_argument('--key_dim', type=int, default=64)
    parser.add_argument('--value_dim', type=int, default=512)
    parser.add_argument('--pix_feat_dim', type=int, default=512)

    # Long-term memory options
    parser.add_argument('--disable_long_term', action='store_true')

    parser.add_argument('--max_mid_term_frames',
                        help='T_max in XMem, decrease to save memory',
                        type=int,
                        default=10)
    parser.add_argument('--min_mid_term_frames',
                        help='T_min in XMem, decrease to save memory',
                        type=int,
                        default=5)
    parser.add_argument('--max_long_term_elements',
                        help='LT_max in XMem, increase if objects disappear for a long time',
                        type=int,
                        default=10000)
    parser.add_argument('--num_prototypes', help='P in XMem', type=int, default=128)

    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--mem_every',
                        help='r in XMem. Increase to improve running speed.',
                        type=int,
                        default=5)
    parser.add_argument(
        '--chunk_size',
        default=1,
        type=int,
        help='''Number of objects to process in parallel as a batch; -1 for unlimited. 
        Set to a small number to save memory.''')

    parser.add_argument(
        '--size',
        default=480,
        type=int,
        help='Resize the shorter side to this size. -1 to use original resolution. ')
    
    return parser


def get_model_and_config(parser: ArgumentParser):
    args = parser.parse_args()
    config = vars(args)
    config['enable_long_term'] = not config['disable_long_term']

    # Load our checkpoint
    network = DEVA(config).cuda().eval()
    if args.devamodel is not None:
        model_weights = torch.load(args.devamodel)
        network.load_weights(model_weights)
    else:
        print('No model loaded.')

    return network, config, args