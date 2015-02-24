# planknet
Planknet is a convolutional neural network for species identification of plankton.  This code has only been tested on Linux and Mac OSX (v10.9).

# pre-installation
Before installing, first install Torch7.  The easiest way to do this is follow the installation guidlines linked to from the "cheatsheet": https://github.com/torch/torch7/wiki/Cheatsheet.  The luarocks installation is extremely fast.  Torch7 should come with most of the necessary packages.  The necessary pacakges are torch, nn, paths, xlua optim.  If any of these are missing, the can be installed from the command line using "luarocks install NAME", where NAME is the name of the package.  One more necessary package is hdf5, which should NOT be installed using luarocks; the luarocks version of hdf5 is not the correct package.  Instead, installed from https://github.com/deepmind/torch-hdf5 (see the usage link at the bottom of the page).  You can easily check which packages are missing by running 'th' from the commandline (which should open up an interactive Lua/Torch session) and typing, for example:  require 'nn'.  If the package is missing, you'll get some kind of error.  If the package is installed, you will get either a 'true' or a table with available functions.

As of now, there is one modification of the nn package that is required in order to run planknet properly:
1) Locate SpatialZeroPadding.lua
2) cp it to SpatialPadding.lua (in the same directory)
3) In the new file, change every instance of "SpatialZeroPadding" to "SpatialPadding"
4) Change the constructor to:
function SpatialPadding:init(pad_l, pad_r, pad_t, pad_b, padding)
  parent.init(self)
  self.pad_l = pad_l
  self.pad_r = pad_r or self.pad_l
  self.pad_t = pad_t or self.pad_l
  self.pad_b = pad_b or self.pad_l
  self.padding = padding or 0
end
5) Change the line "self.output:zero()" -> "self.output:fill(self.padding)"
6) In the same directory is a file called init.lua. To this file add "include('SpatialPadding.lua')".  I don't know if the ordering of the includes matters. I just stuck it after include('SpatialZeroPadding.lua')

You should also install iTorch for future use:  https://github.com/facebook/iTorch

# pre-installation
To install planknet, simply clone it to your local machine.  Inside the planknet directory, you will find the src directory, where all the code is located.  You need to put two more directories into the planknet directory: train and test.  These directories can be downloaded from the Kaggle website:  https://www.kaggle.com/c/datasciencebowl/data.

To test if planknet is working properly on your machine, it is a good idea to train on a subset of the training data.  I usually 'mv train train2' and then 'mkdir train' and copy a few of the training subdirectories back into the new train directory.  Copy over just two or three of the train subdirectories; planknet will only train on the image directories located in train.

# preprocessing
To preprocess the data 'cd src' and run 'th run_preprocessor.lua'.  This will read in all the jpegs from the train subdirectories and assign the appropriate labels.  It also does all additional preprocessing of this data.  It will save the preprocessed data in a directory called preprocessed (alongside src, train and test).  There are many command line options that can be attached to run_preprocessor.  None of these are essential.  You can get the list of options by running something like "th run_preprocessor.lua ?".

# training
To train the neural net, run 'th planknet.lua -maxEpoch #', where # should be replaced with the number of times you want to train on the full dataset.  There are additional commandline options but only -maxEpoch is really essential.

# testing
To run the test data through the net, run 'th run_test.lua'.  The resulting csv file will be saved to a directory called predictions.




