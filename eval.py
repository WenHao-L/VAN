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
"""eval"""

import os
from mindspore import Model
from mindspore import context
from mindspore import nn
from mindspore.common import set_seed

from src.args import args
from src.tools.cell import cast_amp
from src.tools.criterion import get_criterion, NetWithLoss
from src.tools.get_misc import get_dataset, set_device, get_model, pretrained, get_train_one_step
from src.tools.optimizer import get_optimizer

set_seed(args.seed)


def main():
    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    context.set_context(mode=mode[args.graph_mode], device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)
    if args.device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=True)
    set_device(args)

    # get model
    net = get_model(args)
    cast_amp(net)
    criterion = get_criterion(args)

    net_with_loss = NetWithLoss(net, criterion)
    if args.pretrained:
        pretrained(args, net)

    obs_data_url = args.data_url
    args.data_url = '/home/work/user-job-dir/data/'
    if not os.path.exists(args.data_url):
        os.mkdir(args.data_url)
    try:
        import moxing as mox    
        mox.file.copy_parallel(obs_data_url, args.data_url)
        print("Successfully Download {} to {}".format(obs_data_url, args.data_url))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(obs_data_url, args.data_url) + str(e))

    data = get_dataset(args, training=False)
    batch_num = data.val_dataset.get_dataset_size()
    optimizer = get_optimizer(args, net, batch_num)
    # save a yaml file to read to record parameters

    net_with_loss = get_train_one_step(args, net_with_loss, optimizer)
    eval_network = nn.WithEvalCell(net, criterion, args.amp_level in ["O2", "O3", "auto"])
    eval_indexes = [0, 1, 2]
    eval_metrics = {'Loss': nn.Loss(),
                    'Top1-Acc': nn.Top1CategoricalAccuracy(),
                    'Top5-Acc': nn.Top5CategoricalAccuracy()}
    model = Model(net_with_loss, metrics=eval_metrics,
                  eval_network=eval_network,
                  eval_indexes=eval_indexes)
    print(f"=> begin eval")
    results = model.eval(data.val_dataset)
    print(f"=> eval results:{results}")
    print(f"=> eval success")

    obs_result_url = args.result_url
    args.result_url = '/home/work/user-job-dir/result/'
    if not os.path.exists(args.result_url):
        os.mkdir(args.result_url)

    filename = 'result.txt'
    file_path = os.path.join(args.result_url, filename)
    with open(file_path, 'a+') as file:
        file.write(str(results))

    try:
        import moxing as mox
        mox.file.copy_parallel(args.result_url, obs_result_url)
        print("Successfully Upload {} to {}".format(args.result_url, obs_result_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(args.result_url, obs_result_url) + str(e))

if __name__ == '__main__':
    main()
