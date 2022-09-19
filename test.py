from typing import Callable, Optional, List

import cv2
from constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN
from torchtext.legacy.data import Dataset
from batch import Batch
import glob
import os
import torch
import time
from plot_videos import plot_video
from model import Model, build_model
from helpers import load_config
from data import load_data, make_data_iter
from vocabulary import build_vocab, Vocabulary

from translate_sen import *
from konlpy.tag import Okt

def get_latest_checkpoint(ckpt_dir, post_fix="_every" ) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory, of either every validation step or best
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir: directory of checkpoint
    :param post_fixe: type of checkpoint, either "_every" or "_best"

    :return: latest checkpoint file
    """
    # Find all the every validation checkpoints
    list_of_files = glob.glob("{}/*{}.ckpt".format(ckpt_dir,post_fix))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint

def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    print('model path : ', path)

    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location='cuda' if use_cuda else 'cpu')
    return checkpoint


# Validate epoch given a dataset
def validate_on_data(model: Model,
                     data: Dataset,
                     batch_size: int,
                     max_output_length: int,
                     eval_metric: str,
                     loss_function: torch.nn.Module = None,
                     batch_type: str = "sentence",
                     type="val",
                     BT_model=None):

    global train_output

    valid_iter = make_data_iter(
        dataset=data, batch_size=batch_size, batch_type=batch_type,
        shuffle=True, train=False)

    pad_index = model.src_vocab.stoi[PAD_TOKEN]
    # disable dropout
    model.eval()
    # don't track gradients during validation
    # print('trg ? : ', data.trg)
    with torch.no_grad():
        valid_hypotheses = []
        valid_references = []
        valid_inputs = []
        file_paths = []
        all_dtw_scores = []

        valid_loss = 0
        total_ntokens = 0
        total_nseqs = 0

        batches = 0
        for valid_batch in iter(valid_iter):
            # Extract batch
            batch = Batch(torch_batch=valid_batch,
                          pad_index=pad_index,
                          model=model)
            targets = batch.trg
            print('batch src : ', batch.src)
            # print('batch trg : ', batch.trg)
            # print('batch src mask :', batch.src_mask)

            # run as during training with teacher forcing
            if loss_function is not None and batch.trg is not None:
                # Get the loss for this batch
                batch_loss, _ = model.get_loss_for_batch(
                    batch, loss_function=loss_function)

                valid_loss += batch_loss
                total_ntokens += batch.ntokens
                total_nseqs += batch.nseqs

            # If not just count in, run inference to produce translation videos
            if not model.just_count_in:
                # Run batch through the model in an auto-regressive format
                output, attention_scores = model.run_batch(
                    batch=batch,
                    max_output_length=max_output_length)
            # Original output shape
            # print("output_shape_1 : " , output.shape)
            # x10
            # print("target_shape_1 : " , targets.shape)
            # If future prediction
            if model.future_prediction != 0:
                # Cut to only the first frame prediction + add the counter
                # train_output = torch.cat((train_output[:, :, :train_output.shape[2] // (model.future_prediction)], train_output[:, :, -1:]) ,dim=2)
                train_output = torch.cat((output[:, :, :output.shape[2]], output[:, :, -1:]) ,dim=2)
                # Cut to only the first frame prediction + add the counter
                targets = torch.cat((targets[:, :, :targets.shape[2] // (model.future_prediction)], targets[:, :, -1:]),
                                    dim=2)
                # print("output_shape_2 : ", train_output.shape)
                # print("target_shape_2 : ", targets.shape)
                output = train_output

            # For just counter, the inference is the same as GTing
            if model.just_count_in:
                output = train_output

            # Add references, hypotheses and file paths to list
            valid_references.extend(targets)
            valid_hypotheses.extend(output)
            file_paths.extend(batch.file_paths)
            # Add the source sentences to list, by using the model source vocab and batch indices
            valid_inputs.extend([[model.src_vocab.itos[batch.src[i][j]] for j in range(len(batch.src[i]))] for i in
                                 range(len(batch.src))])

            # Calculate the full Dynamic Time Warping score - for evaluation
    return valid_hypotheses

def test(cfg):
    # Load the config file
    cfg = load_config(cfg_file)
    # Load the model directory and checkpoint
    model_dir = cfg["training"]["model_dir"]
    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        ckpt = get_latest_checkpoint(model_dir, post_fix="_best")
        if ckpt is None:
            raise FileNotFoundError("No checkpoint found in directory {}."
                                    .format(model_dir))

    batch_size = cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"])
    batch_type = cfg["training"].get("eval_batch_type", cfg["training"].get("batch_type", "sentence"))
    use_cuda = cfg["training"].get("use_cuda", False)
    eval_metric = cfg["training"]["eval_metric"]
    max_output_length = cfg["training"].get("max_output_length", 300)

    # load the data
    test_data, trg_vocab = load_data(cfg=cfg)
    # To produce testing results
    data_to_predict = {"test": test_data}
    # To produce validation results
    # data_to_predict = {"dev": dev_data}

    # Load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # Build model and load parameters into it
    model = build_model(cfg, src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])
    # If cuda, set model as cuda
    if use_cuda:
        model.cuda()

    # Set up trainer to produce videos
    trainer = TrainManager(model=model, config=cfg, test=True)

    # For each of the required data, produce results
    for data_set_name, data_set in data_to_predict.items():
        # Validate for this data set
        score, loss, references, hypotheses, \
        inputs, all_dtw_scores, file_paths = \
            validate_on_data(
                model=model,
                data=data_set,
                batch_size=batch_size,
                max_output_length=max_output_length,
                eval_metric=eval_metric,
                loss_function=None,
                batch_type=batch_type,
                type="val" if not data_set_name is "train" else "train_inf"
            )
        # json 파일 생성
        trainer.produce_json_files(
            output_joint=hypotheses,
            model_dir=model_dir,
            inputs=inputs,
            type="test"
        )
        # Set which sequences to produce video for
        display = list(range(len(hypotheses)))

        # Produce videos for the produced hypotheses
        trainer.produce_validation_video(
            output_joints=hypotheses,
            inputs=inputs,
            references=references,
            model_dir=model_dir,
            display=display,
            type="test",
            file_paths=file_paths,
        )


def produce_json_files(output_joint, inputs, model_dir):
    import numpy as np
    import json
    rest_pose = [-156.74437241911215, 402.32215884417479, 2480.38551978906, 1
        , -167.79330688968449, 412.89140495859709, 2482.081132917257, 1
        , -178.84745573301871, 423.42553522279962, 2483.667021388695, 1
        , 91.026026522082063, 403.79899182715269, 2473.9967040887391, 1
        , 101.737405679578, 414.57114366465044, 2469.115353234587, 1
        , 112.39844934443732, 425.32082861201133, 2464.1315222745047, 1]

    foot_pose = [123.00766712372553, 436.04672601578204, 2459.046116924681, 1
        , 133.51269789818227, 436.17713957577558, 2455.6184979005238, 1
        , 101.70319787879625, 436.61032555041879, 2468.3398198350756, 1
        , -189.90528657895055, 433.92335357397678, 2485.1432952323075, 1
        , -201.05362760944658, 423.09313804668278, 2487.30617804487, 1
        , -167.72185158101044, 434.24091560947159, 2481.2908232522714, 1]


    gloss_label = '_'.join(inputs)
    gloss_label = gloss_label.replace('</s>', '_')
    gloss_label = gloss_label.replace('<pad>', '_')
    gloss_label = gloss_label.replace('<', '').replace('>', '')
    dir_name = model_dir + '/jsons/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    gloss_dir = model_dir + '/jsons/' + gloss_label + '/'
    if not os.path.exists(gloss_dir):
        os.mkdir(gloss_dir)
    json_load = {}
    joint_dic = {}
    cam = {}
    # pose, hand_r, hand_l = [], [], []
    json_load['version'] = 1.3

    json_load['camparam'] = {'Intrinsics': {
        'data': "2266.79453416316 1.09503716185646 955.347282058466 0 2267.20420101965 570.233341394511 0 0 1"},
                             'CameraMatrix': {'data': "1 0 0 0 0 1 0 0 0 0 1 0"},
                             'Distortion': {'rows': '5',
                                            "data": "-0.100686987342219 0.62252535834329 -0.000521413674628704 -0.00124490857307188 -7.20449461083176"}
                             }

    joint_dic['person_id'] = -1

    for i, skels in enumerate(output_joint):
        # json_load['people'] = joint_dic
        with open(gloss_dir + '/' + str(i).zfill(4) + '.json', 'w', encoding='utf-8') as f:
            # print('skels : ', skels)
            pose = skels[0:39]
            hand_l = skels[39:102]
            hand_r = skels[102:165]

            pose = np.array(pose.cpu()).reshape(13, 3)
            pose[:, 2] = pose[:, 2] * 10000
            pose[:, 0] = pose[:, 0] * pose[:, 2]
            pose[:, 1] = pose[:, 1] * pose[:, 2]
            pose = np.append(pose, np.ones((13, 1)), axis=1)
            pose = pose.reshape(1, -1)

            hand_r = np.array(hand_r.cpu()).reshape(21, 3)
            hand_r[:, 2] = hand_r[:, 2] * 10000
            hand_r[:, 0] = hand_r[:, 0] * hand_r[:, 2]
            hand_r[:, 1] = hand_r[:, 1] * hand_r[:, 2]
            hand_r = np.append(hand_r, np.ones((21, 1)), axis=1)
            hand_r = hand_r.reshape(1, -1)

            hand_l = np.array(hand_l.cpu()).reshape(21, 3)
            hand_l[:, 2] = hand_l[:, 2] * 10000
            hand_l[:, 0] = hand_l[:, 0] * hand_l[:, 2]
            hand_l[:, 1] = hand_l[:, 1] * hand_l[:, 2]
            hand_l = np.append(hand_l, np.ones((21, 1)), axis=1)
            hand_l = hand_l.reshape(1, -1)
            joint_dic['pose_keypoints_3d'] = pose.tolist()[0][0:36] + rest_pose + pose.tolist()[0][
                                                                                  36:52] + foot_pose
            joint_dic['hand_right_keypoints_3d'] = hand_r.tolist()[0]
            joint_dic['hand_left_keypoints_3d'] = hand_l.tolist()[0]
            json_load['people'] = joint_dic
            json.dump(json_load, f, indent='\t')


if __name__ == '__main__':
    import torch
    ################# text2gloss model load ####################
    nature_to_gloss_model = torch.load('/home/climax1001/signlang_test/Models/natural_to_gloss.pt')
    tokenizer = Okt()

    ############# load trained model and vocab #############
    def tokenize_kor(text):
        text = text.replace('(', '').replace(')', '')
        return [text_ for text_ in tokenizer.morphs(text)] 
    
    SRC = data.Field(tokenize=tokenize_kor, init_token='<sos>', eos_token='<eos>', batch_first=True, lower=True)
    TRG = SRC

    train_data = text2glossDataset('train',(SRC, TRG), '/home/climax1001/graduate_project/data/train')
    val_data = text2glossDataset('val',(SRC, TRG), '/home/climax1001/graduate_project/data/val')
    SRC.build_vocab(train_data)
    TRG.build_vocab(train_data)

    on_time = time.time()
    cfg_file = './Models/config.yaml'
    cfg = load_config(cfg_file)
    ckpt = get_latest_checkpoint('./Models', post_fix='_every')
    data_cfg = cfg['data']

    model_checkpoint = load_checkpoint(ckpt, use_cuda=True)
    
    vocab_file = data_cfg['src_vocab']
    src_vocab = Vocabulary(file = vocab_file)
    trg_size = cfg["model"]["trg_size"] + 1

    trg_vocab = [None]*trg_size

    model = build_model(cfg, src_vocab, trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])

    model.future_prediction = 10
    model.cuda()
    model_load_time = time.time() - on_time
    print('model load : ', model_load_time)

    while(1):
        input_glosses = input('텍스트를 입력하시오 \n')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        translation, attention = translate_sentence(input_glosses, SRC, TRG, nature_to_gloss_model, device, logging=True)

        
        # input_glosses = input_glosses.split(' ')
        print('글로스 변환결과 : ', translation[:-1])
        eval_metric = cfg['training']['eval_metric']

        index_num = 0
        start = time.time()
        with open('/home/climax1001/signlang_test/Data/tmp/dev.gloss', 'r') as f:
            i = 0
            while(1):
                gloss = f.readline()
                gloss = tokenizer.morphs(gloss)
                if gloss[:-2] == translation[:-1]:
                    index_num = i
                    print('index_num : ', index_num )
                    break
                    
                elif len(gloss) == 0:
                    print('no match')
                    break
                    
                i += 1

        test_data, _, _ = load_data(cfg=cfg, index_num=index_num)
        print('데이터 로드 시간 : ', time.time() - start)
        # print('src : ' , test_data.examples[0].src)
        # print('trg : ', test_data.examples[0].trg)
        

        # print('model f : ', model.future_prediction)
        # print('model j : ' , model.just_count_in)

        hyphotheses = validate_on_data( model=model,
                                        data=test_data,
                                        batch_size=1,
                                        max_output_length=300,
                                        eval_metric=eval_metric,
                                        loss_function=None,
                                        batch_type="sentence",
                                        type="test")


        print('예측 길이 : ', len(hyphotheses[0]))
        print('예측 시간 : ', time.time() - start)
        plot_video(hyphotheses[0].cpu(), '/home/climax1001/signlang_test/video', 'test1')
        produce_json_files(hyphotheses[0], input_glosses, '/home/climax1001/signlang_test/Models')

        cap = cv2.VideoCapture('video/test1.mp4')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)