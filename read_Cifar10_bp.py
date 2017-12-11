import tables
import pickle
import numpy as np

# Create hdf5 file to store the train, validation and test data
hdf5_path = 'cifar10.hdf5'
data_shape = (0, 32, 32, 3)
img_dtype = tables.Float32Atom()
hdf5_file = tables.open_file(hdf5_path, mode='w')

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f, encoding='latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

train_storage = hdf5_file.create_earray(hdf5_file.root, 'train_img', img_dtype, shape=data_shape)
val_storage = hdf5_file.create_earray(hdf5_file.root, 'val_img', img_dtype, shape=data_shape)
test_storage = hdf5_file.create_earray(hdf5_file.root, 'test_img', img_dtype, shape=data_shape)

# Read train data
train_labels = np.array([])
for i in range(1,5):
    x ,y = load_CIFAR_batch("./cifar-10-batches-py/data_batch_%d" %i)
    for i in range(len(x)):
        img = x[i]
        train_storage.append(img[None])
    train_labels = np.append(train_labels,y).astype(int)
hdf5_file.create_array(hdf5_file.root, 'train_labels', train_labels)

# Read validation data
x ,y = load_CIFAR_batch("./cifar-10-batches-py/data_batch_5")
for i in range(len(x)):
    img = x[i]
    val_storage.append(img[None])
val_labels = y
hdf5_file.create_array(hdf5_file.root, 'val_labels', val_labels)

# Read test data
x ,y = load_CIFAR_batch("./cifar-10-batches-py/test_batch")
for i in range(len(x)):
    img = x[i]
    test_storage.append(img[None])
test_labels = y
hdf5_file.create_array(hdf5_file.root, 'test_labels', test_labels)

hdf5_file.close()

