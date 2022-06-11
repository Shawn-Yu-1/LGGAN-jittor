"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import jittor as jt
import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()

jt.flags.use_cuda = (jt.has_cuda and opt.gpu_ids != "-1")

dataset = data.create_dataloader(opt)
dataloader = dataset().set_attrs(batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads),
        drop_last=opt.isTrain)

dataloader.initialize(opt)
print("the testDataset is contain %d labels" %(len(dataloader)))

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    # generated, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
    #            label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
    #            label_3_21,label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, label_3_29, label_3_30, label_3_31, \
    #            label_3_32, label_3_33, label_3_34, result_0, result_1, result_2, result_3, result_4,result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
    #            result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
    #            result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28 , result_29 , result_30, \
    #            result_31 , result_32 , result_33 , result_34,feature_score, target, index,  attention_global, attention_local, = model(data_i, mode='inference')

    generated, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
    label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
    label_3_21, label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, \
    result_0, result_1, result_2, result_3, result_4, result_5, result_6, result_7, result_8, result_9, result_10, \
    result_11, result_12, result_13, result_14, result_15, result_16, result_17, result_18, result_19, result_20, \
    result_21, result_22, result_23, result_24, result_25, result_26, result_27, result_28,  feature_score, \
    target, index, attention_global, attention_local = model(data_i, mode='inference')

    # attention_global= (attention_global - 0.5)/0.5
    # attention_local= (attention_local - 0.5)/0.5
    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),

                               ('global_image', result_global[b]),
                               ('local_image', result_local[b]),
                               ('global_attention', 1 - attention_local[b]/jt.max(attention_local[b]).item() ),
                               ('local_attention', attention_local[b]/jt.max(attention_local[b]).item() ),
                               ('synthesized_image', generated[b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()
