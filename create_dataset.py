from PIL import Image
import numpy as np
import os

cur_path = os.getcwd()
classes = 43

def create_training_data(state):
    #data = np.empty(shape=[0,1])
    label = np.empty(shape = [0,1])
    for p in range(classes):
        path = os.path.join(cur_path, state, str(p))
        images = os.listdir(path)
        for i in images:
            try:
                img = Image.open(path+'\\'+i)
                img = img.resize((30,30))
                img = np.array(img)
                data = np.append(data, [img])
                label = np.append(label, [p])
            except Exception as e:
                print(str(e))

    data_name = 'data_'+ state + '.npy'
    label_name = 'label_'+ state + '.npy'
    np.save(data_name, data)
    np.save(label_name, label)

create_training_data('Train')
create_training_data('Test')
