import class_names

imagenet_v2_folders = ['datasets/imagenetv2-top-images-format-val', 'datasets/imagenetv2-threshold0.7-format-val', 'datasets/imagenetv2-matched-frequency-format-val']
for folder in imagenet_v2_folders:
    for name in (os.listdir('/home/davidva/' + folder)):
        os.rename('/home/davidva/' + folder + '/'  + name, '/home/davidva/' + folder + '/'  + class_names.all_wnids[int(name)])