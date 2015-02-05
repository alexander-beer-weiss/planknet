
require 'torch'
require 'image'  -- decode jpegs
require 'paths'  -- read OS directory structure
require 'hdf5'  -- read and write to hdf5 format


preprocessed_path = '../preprocessed'


dofile 'load_data.lua'

dofile 'preprocess_data.lua'

dofile 'save_preprocessed.lua'


