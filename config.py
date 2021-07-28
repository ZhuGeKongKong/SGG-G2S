"""
Configuration file!
"""
import os
from argparse import ArgumentParser
import numpy as np

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = '/hhd/***/datasets/visual_genome/data/genome/'

def path(fn):
    return os.path.join(DATA_PATH, fn)

def stanford_path_1(fn):
    return os.path.join(DATA_PATH, 'stanford_filtered', fn)

def stanford_path(fn):
    return os.path.join(DATA_PATH, fn)
# =============================================================================
# Update these with where your data is stored ~~~~~~~~~~~~~~~~~~~~~~~~~

VG_IMAGES = '/hhd/***/datasets/visual_genome/data/genome/VG_100K'
RCNN_CHECKPOINT_FN = path('faster_rcnn_500k.h5')

IM_DATA_FN = stanford_path('image_data.json')
VG_SGG_FN = stanford_path('VG-SGG.h5')
VG_SGG_DICT_FN = stanford_path('VG-SGG-dicts.json')
PROPOSAL_FN = stanford_path('proposals.h5')

COCO_PATH = '/mnt/data1/fankaixuan/dataset/mscoco/'
# =============================================================================
# =============================================================================


MODES = ('sgdet', 'sgcls', 'predcls', 'predcls_nongtbox', 'detclass')
POS_GCN = ('parall','stack', 'stack_in','front')

BOX_SCALE = 1024  # Scale at which we have the boxes
IM_SCALE = 592      # Our images will be resized to this res without padding

# Proposal assignments
BG_THRESH_HI = 0.5
BG_THRESH_LO = 0.0

RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
RPN_NEGATIVE_OVERLAP = 0.3

# Max number of foreground examples
RPN_FG_FRACTION = 0.5
FG_FRACTION = 0.25
# Total number of examples
RPN_BATCHSIZE = 256
ROIS_PER_IMG = 256
REL_FG_FRACTION = 0.25
RELS_PER_IMG = 256

RELS_PER_IMG_REFINE = 64

BATCHNORM_MOMENTUM = 0.01
ANCHOR_SIZE = 16

ANCHOR_RATIOS = (0.23232838, 0.63365731, 1.28478321, 3.15089189) #(0.5, 1, 2)
ANCHOR_SCALES = (2.22152954, 4.12315647, 7.21692515, 12.60263013, 22.7102731) #(4, 8, 16, 32)

class ModelConfig(object):
    """Wrapper class for model hyperparameters."""
    def __init__(self):
        """
        Defaults
        """
        self.coco = None
        self.ckpt = None
        self.save_dir = None
        self.lr = None
        self.batch_size = None
        self.val_size = None
        self.l2 = None
        self.clip = None
        self.num_gpus = None
        self.num_workers = None
        self.print_interval = None
        self.gt_box = None
        self.mode = None
        self.refine = None
        self.ad3 = False
        self.test = False
        self.adam = False
        self.multi_pred=False
        self.cache = None
        self.model = None
        self.use_proposals=False
        self.use_resnet=False
        self.use_tanh=False
        self.use_bias = False
        self.limit_vision=False
        self.num_epochs=None
        self.old_feats=False
        self.order=None
        self.det_ckpt=None
        self.nl_edge=None
        self.nl_obj=None
        self.nh_edge=None
        self.nh_obj=None
        self.nl_obj_gcn = None
        self.hidden_dim=None
        self.attention_dim=None
        self.pass_in_obj_feats_to_decoder = None
        self.pass_in_obj_feats_to_edge = None
        self.pass_in_obj_feats_to_gcn = None
        self.pass_embed_togcn = None
        self.pooling_dim = None
        self.rec_dropout = None
        self.att_alpha_c = None
        self.nl_adj = None
        self.with_adj_mat = False
        self.adj_embed_dim = None
        self.bg_num_graph = -1
        self.bg_num_rel = -1
        self.adj_embed=None
        self.mean_union_feat=None,
        self.ch_res=None,
        self.with_att=None,
        self.fb_thr = None
        self.with_gcn = None
        self.gcn_adj_type = None
        self.with_biliner_score = None
        self.where_gcn=None
        self.gt_adj_mat = None
        self.type_gcn = None
        self.edge_ctx_type = None
        self.nms_union = None
        self.cosine_dis = False
        self.test_alpha = None
        self.fpn_on = False
        self.debug = False
        self.neg_time = 0
        self.keep_pred = 'all'
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())

        print("~~~~~~~~ Hyperparameters used: ~~~~~~~")
        for x, y in self.args.items():
            print("{} : {}".format(x, y))

        self.__dict__.update(self.args)
        
        if len(self.ckpt) != 0:
            self.ckpt = os.path.join(ROOT_PATH, self.ckpt)
        else:
            self.ckpt = None

        if len(self.cache) != 0:
            self.cache = os.path.join(ROOT_PATH, self.cache)
        else:
            self.cache = None

        if len(self.save_dir) == 0:
            self.save_dir = None
        else:
            self.save_dir = os.path.join(ROOT_PATH, self.save_dir)
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

        assert self.val_size >= 0

        if self.mode not in MODES:
            raise ValueError("Invalid mode: mode must be in {}".format(MODES))

        if self.where_gcn not in POS_GCN:
            raise ValueError("Invalid mode: mode must be in {}".format(POS_GCN))

        if self.model not in ('motifnet', 'stanford'):
            raise ValueError("Invalid model {}".format(self.model))

        print('--------',self.ckpt)
        if self.ckpt is not None and not os.path.exists(self.ckpt):
            
            raise ValueError("Ckpt file ({}) doesnt exist".format(self.ckpt))
        
        
    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')


        # Options to deprecate
        parser.add_argument('-coco', dest='coco', help='Use COCO (default to VG)', action='store_true')
        parser.add_argument('-ckpt', dest='ckpt', help='Filename to load from', type=str, default='')
        parser.add_argument('-det_ckpt', dest='det_ckpt', help='Filename to load detection parameters from', type=str, default='')

        parser.add_argument('-save_dir', dest='save_dir',
                            help='Directory to save things to, such as checkpoints/save', default='', type=str)

        parser.add_argument('-ngpu', dest='num_gpus', help='cuantos GPUs tienes', type=int, default=3)
        parser.add_argument('-nwork', dest='num_workers', help='num processes to use as workers', type=int, default=1)

        parser.add_argument('-lr', dest='lr', help='learning rate', type=float, default=1e-3)
        parser.add_argument('-fb_thr', dest='fb_thr', help='the threshold of foreground or background!', type=float, default=0.5)
        parser.add_argument('-att_alpha_c', dest='att_alpha_c', help='learning rate', type=float, default=0.70602)
        parser.add_argument('-b', dest='batch_size', help='batch size per GPU',type=int, default=2)
        parser.add_argument('-val_size', dest='val_size', help='val size to use (if 0 we wont use val)', type=int, default=5000)

        parser.add_argument('-l2', dest='l2', help='weight decay', type=float, default=1e-4)
        parser.add_argument('-clip', dest='clip', help='gradients will be clipped to have norm less than this', type=float, default=5.0)
        parser.add_argument('-p', dest='print_interval', help='print during training', type=int,
                            default=100)
        parser.add_argument('-m', dest='mode', help='mode \in {sgdet, sgcls, predcls, detclass}', type=str,
                            default='sgdet')
        parser.add_argument('-model', dest='model', help='which model to use? (motifnet, stanford). If you want to use the baseline (NoContext) model, then pass in motifnet here, and nl_obj, nl_edge=0', type=str,
                            default='motifnet')
        parser.add_argument('-old_feats', dest='old_feats', help='Use the original image features for the edges', action='store_true')
        parser.add_argument('-order', dest='order', help='Linearization order for Rois (confidence -default, size, random)',
                            type=str, default='confidence')
        parser.add_argument('-cache', dest='cache', help='where should we cache predictions', type=str,
                            default='')
        parser.add_argument('-gt_box', dest='gt_box', help='use gt boxes during training', action='store_true')
        parser.add_argument('-adam', dest='adam', help='use adam. Not recommended', action='store_true')
        parser.add_argument('-test', dest='test', help='test set', action='store_true')
        parser.add_argument('-multipred', dest='multi_pred', help='Allow multiple predicates per pair of box0, box1.', action='store_true')
        parser.add_argument('-nepoch', dest='num_epochs', help='Number of epochs to train the model for',type=int, default=25)
        parser.add_argument('-resnet', dest='use_resnet', help='use resnet instead of VGG', action='store_true')
        parser.add_argument('-with_adj_mat', dest='with_adj_mat', help='add graph loss', action='store_true')
        parser.add_argument('-proposals', dest='use_proposals', help='Use Xu et als proposals', action='store_true')
        parser.add_argument('-nl_obj', dest='nl_obj', help='Num object layers', type=int, default=1)
        parser.add_argument('-nl_obj_gcn', dest='nl_obj_gcn', help='Num object gcn layers', type=int, default=1)
        parser.add_argument('-nl_edge', dest='nl_edge', help='Num edge layers', type=int, default=2)
        parser.add_argument('-nh_obj', dest='nh_obj', help='Num edge heads', type=int, default=0)
        parser.add_argument('-nh_edge', dest='nh_edge', help='Num edge heads', type=int, default=0)
        parser.add_argument('-nl_adj', dest='nl_adj', help='Num adj layers', type=int, default=0)
        parser.add_argument('-hidden_dim', dest='hidden_dim', help='Num edge layers', type=int, default=256)
        parser.add_argument('-attention_dim', dest='attention_dim', help='the dimension of attention layer', type=int, default=256)
        parser.add_argument('-pooling_dim', dest='pooling_dim', help='Dimension of pooling', type=int, default=4096)
        parser.add_argument('-adj_embed_dim', dest='adj_embed_dim', help='Dimension of pooling', type=int, default=4096)
        parser.add_argument('-pass_in_obj_feats_to_decoder', dest='pass_in_obj_feats_to_decoder', action='store_true')
        parser.add_argument('-pass_in_obj_feats_to_edge', dest='pass_in_obj_feats_to_edge', action='store_true')
        parser.add_argument('-pass_in_obj_feats_to_gcn', dest='pass_in_obj_feats_to_gcn', action='store_true')
        parser.add_argument('-pass_embed_togcn', dest='pass_embed_togcn', action='store_true')
        parser.add_argument('-rec_dropout', dest='rec_dropout', help='recurrent dropout to add', type=float, default=0.1)
        parser.add_argument('-use_bias', dest='use_bias',  action='store_true')
        parser.add_argument('-use_tanh', dest='use_tanh',  action='store_true')
        parser.add_argument('-adj_embed', dest='adj_embed',  action='store_true')
        parser.add_argument('-mean_union_feat', dest='mean_union_feat',  action='store_true')
        parser.add_argument('-limit_vision', dest='limit_vision',  action='store_true')
        parser.add_argument('-ch_res', dest='ch_res',  action='store_true')
        parser.add_argument('-with_att', dest='with_att', action='store_true')
        parser.add_argument('-with_gcn', dest='with_gcn', action='store_true')
        parser.add_argument('-type_gcn', dest='type_gcn', help='which type gcn used?', type=str, default='normal')
        parser.add_argument('-gcn_adj_type', dest='gcn_adj_type', help='soft? hard?', type=str, default='hard')
        parser.add_argument('-bg_num_graph',  dest='bg_num_graph', help='the number of background need been counted, when compute graph loss, -1 is all, -2 is half',
                            type=int, default=-1)
        parser.add_argument('-bg_num_rel', dest='bg_num_rel',
                            help='the number of background need been counted, when compute relatinship loss, -1 is all, -2 is half',
                            type=int, default=-1)
        parser.add_argument('-neg_time', dest='neg_time', type=int, default=0)
        parser.add_argument('-with_biliner_score', dest='with_biliner_score', action='store_true')
        parser.add_argument('-where_gcn', dest='where_gcn',help='parall? or stack?', type=str, default='parall')
        parser.add_argument('-edge_ctx_type', dest='edge_ctx_type', help='obj? or union?', type=str, default='obj')
        parser.add_argument('-gt_adj_mat', dest='gt_adj_mat', action='store_true')
        parser.add_argument('-nms_union', dest='nms_union', action='store_true')
        parser.add_argument('-cosine_dis', dest='cosine_dis', action='store_true')
        parser.add_argument('-test_alpha', dest='test_alpha', help='test_alpha', type=float, default=0.5)
        parser.add_argument('-fpn_on', dest='fpn_on', action='store_true')
        parser.add_argument('-keep_pred', dest='keep_pred', help='all? father? child?', type=str, default='all')
        parser.add_argument('-debug', dest='debug', action='store_true')
        return parser
