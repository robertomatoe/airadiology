import os
from pathlib import Path
import shutil
import random

origin_path = './data/origin_w_label'
output_data_path = './data/custom_data'

cnt_w_lable = 0
cnt_wo_lable = 0
cnt_train = 0
cnt_valid = 0

empty_file_sample_ratio = 0.01
valid_ratio = 0.1

if os.path.isdir(output_data_path):
    shutil.rmtree(output_data_path)

Path(output_data_path).mkdir(parents=True, exist_ok=True)

classes_txt = open(f'{output_data_path}/classes.txt', 'w')
classes_txt.write('tumor\n')
classes_txt.close()

patient_list = list(filter(lambda x: x.startswith('Patient'), os.listdir(origin_path)))

for patient in patient_list:
    file_list = os.listdir(f'{origin_path}/{patient}')

    for file in file_list:
        src = f'{origin_path}/{patient}/{file}'

        if file == 'classes.txt':
            continue
        elif '.txt' in file:
            fr = open(src, 'r')
            origin_line = fr.readline()
            origin_line_word = origin_line.split(' ')
            if origin_line_word[0] == '2':
                origin_line_word[0] = '0'
            ret_line = ' '.join(origin_line_word)

            anno_dest = f'{output_data_path}/{patient}_{file}'
            fw = open(anno_dest, 'w')
            fw.write(ret_line)
            fr.close()
            fw.close()
            cnt_w_lable += 1
        
        elif '.jpg' in file:
            # Empty File gen.
            if not os.path.isfile(f'{src}'.replace('.jpg', '.txt')):
                # Empty File Sampling
                if random.random() > empty_file_sample_ratio:
                    continue
                anno_dest = f'{output_data_path}/{patient}_{file}'.replace('.jpg', '.txt')
                fw = open(anno_dest, 'w')
                fw.close()
                cnt_wo_lable += 1

            
            img_dest = f'{output_data_path}/{patient}_{file}'
            shutil.copy(src, img_dest)


print("[DONE]")
print(f"Images with label: {cnt_w_lable}")
print(f"Images without label: {cnt_wo_lable}")
print(f"Total Images: {cnt_wo_lable + cnt_w_lable}")

f_info = open(f'{output_data_path}/data_info.txt', 'w')
f_info.write(f"Images with label: {cnt_w_lable}")
f_info.write(f"Images without label: {cnt_wo_lable}")
f_info.write(f"Total Images: {cnt_wo_lable + cnt_w_lable}")
f_info.close
