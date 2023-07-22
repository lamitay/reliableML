# Opening the file that maps between file name to class number
f = open('/home/davidva/datasets/imagenet_vid_ytbb_robust/imagenet-vid-robust/metadata/labels.json')
data = json.load(f)
# lens_list = []

# for obj in data:
#     print(data[obj})
#     break

# for key in data.keys():
#     lens_list.append(len(data[key]))
# len(data['val/ILSVRC2015_val_00000003/000237.JPEG'])

# 1: 20665, 2: 1463, 3: 51 is the number of occurrences of each number of labels that the images hace




import shutil

all_data_folder = '/home/davidva/datasets/imagenet_vid_ytbb_robust/imagenet-vid-robust/val/'

# the next loops should be run only once:
# create folder whose names are 0, 1, 2... 29. One folder for each class number
for i in range(0,30):
    os.makedirs('/home/davidva/datasets/imagenet_vid_ytbb_robust/imagenet-vid-robust/val/' + str(i))

# create a dictionary whose keys are the folder names and values are the class number(s) of the folder
folder_name_to_class_number = {}
for file_name in data:
    folder_name = file_name[:-12]
    folder_name_to_class_number[folder_name] = data[file_name]

# create a list of all videos/folders
folder_names = os.listdir(all_data_folder)
for i in range(0, 30):
    folder_names.remove(str(i))

# move all the images from the original 1100+ folders to the 30 new folders
for folder_name in folder_names:
    folder_class = folder_name_to_class_number['val/' + folder_name]
    files_in_folder = os.listdir(all_data_folder + folder_name)
    for file_name in files_in_folder:
    # construct full file path
        source = all_data_folder + folder_name + '/' + file_name
        destination_folder = all_data_folder + str(folder_class[0]) + '/'
        destination = destination_folder + file_name
        # move only files
        shutil.move(source, destination)

# remove all empty folders from the data folder
for folder_name in folder_names:
    try:
        shutil.rmtree(all_data_folder + folder_name)
    except:  # if you run the loop once it will never get to the exception
        continue




h = open('/home/davidva/datasets/imagenet_vid_ytbb_robust/imagenet-vid-robust/misc/wnid_map.json')
imagenet_class_to_vid_robust_class = json.load(h)
for i in range(0, 400):
    try:
        print(i, imagenet_class_to_vid_robust_class[class_names.all_wnids[i]])
        break
    except:
        continue

k = open('/home/davidva/datasets/imagenet_vid_ytbb_robust/imagenet-vid-robust/misc/imagenet_vid_class_index.json')
vid_robust_class_num_to_class_name = json.load(k)
vid_robust_class_name_to_class_num = {v[0]: k for k, v in vid_robust_class_num_to_class_name.items()}
for i in range(0, 400):
    try:
        print(i, vid_robust_class_name_to_class_num[imagenet_class_to_vid_robust_class[class_names.all_wnids[i]]])
        break
    except:
        continue






import class_names
imagenet_v2_folders = ['datasets/imagenetv2-top-images-format-val', 'datasets/imagenetv2-threshold0.7-format-val', 'datasets/imagenetv2-matched-frequency-format-val']
for folder in imagenet_v2_folders:
    for name in (os.listdir('/home/davidva/' + folder)):
        os.rename('/home/davidva/' + folder + '/'  + name, '/home/davidva/' + folder + '/'  + class_names.all_wnids[int(name)])