# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train"""
from ast import arg
import os

from mindspore import Model
from mindspore import context
from mindspore import nn
from mindspore.common import set_seed
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from src.args import args
from src.tools.callback import EvaluateCallBack
from src.tools.cell import cast_amp
from src.tools.criterion import get_criterion, NetWithLoss
from src.tools.get_misc import get_dataset, set_device, get_model, pretrained, get_train_one_step
from src.tools.optimizer import get_optimizer


def sync_data(args, environment="train"):
    if environment == "train":
        workroot = "/home/work/user-job-dir"
    elif environment == "debug":
        workroot = "/home/ma-user/work/"

    data_dir = os.path.join(workroot, "data")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    train_dir = os.path.join(workroot, "model")
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    if environment == 'train':
        obs_data_url = args.data_url
        args.data_url = data_dir

        try:
            import moxing as mox
            mox.file.copy_parallel(obs_data_url, data_dir) 
            print("Successfully Download {} to {}".format(obs_data_url, args.data_url))
        except Exception as e:
            print('moxing download {} to {} failed: '.format(obs_data_url, args.data_url) + str(e))
    
    return train_dir


def sync_model(args, environment="train"):
    if environment == "train":
        workroot = "/home/work/user-job-dir"
    elif environment == "debug":
        workroot = "/home/ma-user/work/"

    train_dir = os.path.join(workroot, "model")
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    if environment == 'train':
        obs_train_url = args.train_url
        args.train_url = train_dir

        try:
            import moxing as mox
            mox.file.copy_parallel(args.train_url, obs_train_url)
            print("Successfully Upload {} to {}".format(args.train_url, obs_train_url))
        except Exception as e:
            print('moxing upload {} to {} failed: '.format(args.train_url, obs_train_url) + str(e))


def main():
    assert args.crop, f"{args.arch} is only for evaluation"
    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    context.set_context(mode=mode[args.graph_mode], device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)
    if args.device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=True)
    rank = set_device(args)
    set_seed(args.seed + rank)

    # get model and cast amp_level
    net = get_model(args)
    cast_amp(net)
    criterion = get_criterion(args)
    net_with_loss = NetWithLoss(net, criterion)
    if args.pretrained:
        pretrained(args, net)

    if args.run_modelarts:
        train_dir = "/cache/ckpt_" + str(rank)
    elif args.run_intelligent:
        train_dir = os.path.join("cache", "output", "ckpt_{}".format(rank))
        data_dir = os.path.join("cache", "dataset")
        args.data_url = data_dir
    elif args.run_openi:
        train_dir = sync_data(args, environment="train")
        train_dir = os.path.join(train_dir, "ckpt_{}".format(rank))

    data = get_dataset(args)
    batch_num = data.train_dataset.get_dataset_size()
    optimizer = get_optimizer(args, net, batch_num)
    # save a yaml file to read to record parameters

    net_with_loss = get_train_one_step(args, net_with_loss, optimizer)

    eval_network = nn.WithEvalCell(net, criterion, args.amp_level in ["O2", "O3", "auto"])
    eval_indexes = [0, 1, 2]
    model = Model(net_with_loss, metrics={"acc", "loss"},
                  eval_network=eval_network,
                  eval_indexes=eval_indexes)

    config_ck = CheckpointConfig(save_checkpoint_steps=data.train_dataset.get_dataset_size()*10,
                                 keep_checkpoint_max=args.save_every)
    time_cb = TimeMonitor(data_size=data.train_dataset.get_dataset_size())

    ckpoint_cb = ModelCheckpoint(prefix=args.arch + str(rank), directory=train_dir,
                                 config=config_ck)
    loss_cb = LossMonitor(per_print_times=data.train_dataset.get_dataset_size())
    eval_cb = EvaluateCallBack(model, eval_dataset=data.val_dataset, src_url=train_dir,
                               train_url=os.path.join(args.train_url, "ckpt_" + str(rank)),
                               save_freq=args.save_every)

    print("begin train")
    model.train(int(args.epochs - args.start_epoch), data.train_dataset,
                callbacks=[time_cb, ckpoint_cb, loss_cb, eval_cb],
                dataset_sink_mode=args.dataset_sink_mode)
    print("train success")

    if not args.run_intelligent:
        if args.run_openi:
            sync_model(args)
        elif args.run_modelarts:
            import moxing as mox
            mox.file.copy_parallel(src_url=train_dir, dst_url=os.path.join(args.train_url, "ckpt_" + str(rank)))


if __name__ == '__main__':
    main()
