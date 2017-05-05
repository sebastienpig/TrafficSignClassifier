
Network parameters
n_input = 32 * 32 * 3
nb_filters = 32
kernel_size = (3, 3)
input_shape = (32, 32, 3)
n_fc1 = 512
n_fc2 = 128
in_channels = 3
pool_size = 2 # i.e. (2,2)

dropout_conv = 0.9
dropout_fc = 0.9

weights_stddev = 0.1
weights_mean = 0.0
biases_mean = 0.0

padding = 'VALID'
if padding == 'SAME':
conv_output_length = 6
elif padding == 'VALID':
conv_output_length = 5
else:
raiseException("Unknown padding.")



### Dataset Exploration

####  Summary of the data set.

Exploratory Visualization of the dataset

The training and test sets are provided as a form of pickle data.

Pickle  “serialises” the object first before writing it to file. Pickling is a way to convert a python object (list, dict, etc. ) into a character stream. This character stream contains all the information necessary to reconstruct the object.


<pre>

training_file = "traffic-signs-data/train.p"
testing_file = "traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

</pre># SignClassification
