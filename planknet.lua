
require 'torch'
require 'nn'
require 'xlua'  -- progress bar
require 'optim'  -- confusion matrix; gradient decent optimization
require 'image'  -- decode jpegs
require 'paths'  -- read OS directory structure
require 'hdf5'  -- write and read to hdf5 format




dofile 'read_cmdLine.lua'  -- parse command line options, store in opt

dofile 'helper_functions.lua'  -- maxdims, pad, table_shuffle

dofile 'split_data.lua'  -- read in preprocessed data and splits into training + cross-validation

dofile 'build_convnet.lua'  -- convnet = nn.Sequential()

dofile 'train_prep.lua'  -- defines confusion matrix, stores initial weights/gradients

dofile 'config_optimizer.lua'  -- optim.sgd, optim.asgd, optim.lbfgs, optim.cg 

dofile 'train.lua'  -- train convnet

dofile 'cross_validate.lua'  -- test convnet with cross-validation data

dofile 'go_plankton.lua'  -- goplankton() : loop calls training and crossvalidation