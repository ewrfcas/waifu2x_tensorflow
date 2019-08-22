from waifu2x import upconv7, vgg7
import os

test_imgs = ['data/face1.jpg', 'data/face2.jpg']

upconv7_model0 = upconv7('models/upconv_7/noise0_scale2.0x_model.json')
upconv7_model1 = upconv7('models/upconv_7/noise1_scale2.0x_model.json')
upconv7_model2 = upconv7('models/upconv_7/noise2_scale2.0x_model.json')
upconv7_model3 = upconv7('models/upconv_7/noise3_scale2.0x_model.json')

vgg7_model0 = vgg7('models/vgg_7/noise0_model.json')
vgg7_model1 = vgg7('models/vgg_7/noise1_model.json')
vgg7_model2 = vgg7('models/vgg_7/noise2_model.json')
vgg7_model3 = vgg7('models/vgg_7/noise3_model.json')

for img_path in test_imgs:
    upconv7_model0.up_scale(img_path, output_path=os.path.join('output', 'upconv7_0_' + img_path.split('/')[-1]))
    upconv7_model1.up_scale(img_path, output_path=os.path.join('output', 'upconv7_1_' + img_path.split('/')[-1]))
    upconv7_model2.up_scale(img_path, output_path=os.path.join('output', 'upconv7_2_' + img_path.split('/')[-1]))
    upconv7_model3.up_scale(img_path, output_path=os.path.join('output', 'upconv7_3_' + img_path.split('/')[-1]))

    vgg7_model0.up_scale(img_path, output_path=os.path.join('output', 'vgg7_0_' + img_path.split('/')[-1]))
    vgg7_model1.up_scale(img_path, output_path=os.path.join('output', 'vgg7_1_' + img_path.split('/')[-1]))
    vgg7_model2.up_scale(img_path, output_path=os.path.join('output', 'vgg7_2_' + img_path.split('/')[-1]))
    vgg7_model3.up_scale(img_path, output_path=os.path.join('output', 'vgg7_3_' + img_path.split('/')[-1]))
