
require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

require 'nn'
require 'image'  -- decode jpegs
require 'paths'  -- read OS directory structure
require 'hdf5'  -- read and write to hdf5 format


images_path = '../train'
preprocessed_path = '../preprocessed'


dofile 'helper_functions.lua'

dofile 'load_data.lua'

-- dofile 'preprocess_data.lua'

dofile 'save_preprocessed.lua'


