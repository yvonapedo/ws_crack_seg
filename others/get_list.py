
def do():
    # img_path = r'C:\Users\yvona\Documents\NPU_research\research\SSVS\datasets\special\testA_128\good'
    img_path = r'C:\Users\yvona\Downloads\RIPU-main\RIPU-main\datasets\cracktree\image'
    for split in os.listdir(img_path):
        file_name = os.path.splitext(split)

        with open('test.txt', 'a') as f:
            # f.write('/media/tao/新加卷/PITE/SSVS-main/datasets/crack/testB/'+file_name[0] +'.jpg'+' '+'/media/tao/新加卷/PITE/SSVS-main/datasets/crack/testA/'+file_name[0] + '.bmp'+'\n')
            f.write('C:/Users/yvona/Downloads/cracktree/gt/'+file_name[0] +'.jpg' +'\n')


import os




def rename():
    img_path = '/media/tao/新加卷/HZW/data/chapter4_weakly'
    for split in os.listdir(img_path):
        old_path = os.path.join(img_path,split)
        file_name = os.path.splitext(split)
        file_name = '1' + file_name[0]
        new_path = os.path.join(img_path,file_name) + '.png'
        os.rename(old_path,new_path)
if __name__ == '__main__':
    do()

