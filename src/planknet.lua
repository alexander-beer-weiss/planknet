
require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

require 'nn'
require 'xlua'  -- progress bar
require 'optim'  -- confusion matrix; gradient decent optimization
require 'paths'  -- read OS directory structure

dofile 'hdf5io.lua'
dofile 'convnet.lua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Training/Optimization')
cmd:text()
cmd:text('Options:')
cmd:option('-trainingdir', '../train', 'location of training directory')
cmd:option('-testingdir', '../test', 'location of testing directory')
cmd:option('-misclassifieddir', '../misclassdata_path', 'location of misclassified directory')
cmd:option('-preprocesseddir', '../preprocessed', 'location of preprocessed directory')
cmd:option('-netDatadir', '../NNsave', 'location of neural net data directory')
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-visualize', false, 'visualize input data and weights during training')
cmd:option('-plot', false, 'live plot')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-1, 'learning rate at t=0')  -- could make this variable.  Start big and decay.
cmd:option('-batchSize', 10, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0.0, 'weight decay (SGD only)')
cmd:option('-transfer', 'ReLU', 'transfer function: Tanh | ReLU | Sigmoid')
cmd:option('-dropout', '0,0.2,0.2', 'fraction of connections to drop: comma seperated numbers in the range 0 to 1')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 4, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-maxEpoch', 20, 'maximum number of epochs during training')  -- set to -1 for unlimited epochs
cmd:option('-loadNet', '', 'load from previous opt')
cmd:option('-learningRateScale', 0.5, 'factor to reduce the learning rate if the score is increasing')
cmd:text()
opt = cmd:parse(arg or {})

local dropout = {}
for match in opt.dropout:gmatch("([%d%.%+%-]+),?") do
  table.insert(dropout,tonumber(match))
end
opt.dropout = dropout

h5=hdf5IO(opt)
h5:loadFromHDF5()

plankton_images_train = {}  -- images of plankton ( array with values of type torch.Tensor(num_channels, num_rows, num_cols) )
plankton_images_cv = {}  -- set aside 20% for cross-validation
plankton_targets_train = {}  -- target labels for images  ( array with values of type string )
plankton_targets_cv = {}  -- set aside (same) 20% for cross-validation
plankton_paths_train = {}  -- file names, so we can visualize which ones our neural net misses
plankton_paths_cv={}


--image_files, image_targets, image_paths = table_shuffle(image_files, image_targets, image_paths)
local offset_index = 0
for i = 1,#species_count do
  local shuffle = torch.randperm(species_count[i])
  for j = 1, warped_multiples[i] do
    for k = 1,math.floor(species_count[i]*0.8) do -- 80% of examples used for training
      table.insert(plankton_images_train,image_files[ offset_index + shuffle[k] ])  -- need to shuffle with each species first...
      table.insert(plankton_targets_train,image_targets[ offset_index + shuffle[k] ])
      table.insert(plankton_paths_train,image_paths[ offset_index + shuffle[k] ])
    end
    if j == 1 then  -- We only extract CV images from the unwarped dataset
      for k = math.floor(species_count[i]*0.8)+1,species_count[i] do  -- 20% of (unwarped) examples used for cross-validation
        table.insert(plankton_images_cv,image_files[ offset_index + shuffle[k] ])
        table.insert(plankton_targets_cv,image_targets[ offset_index + shuffle[k] ])
        table.insert(plankton_paths_cv,image_paths[ offset_index + shuffle[k] ])
      end
    end
    offset_index = offset_index + species_count[i]
  end
end

-- store dimensions of images (note: all images are the same size due to padding)
height = image_files[1]:size(2)
width = image_files[1]:size(3)
opt.height=height
opt.width=width

print ('image size: ' .. #image_files,opt.height,opt.width)
-- classes
plankton_ids={}
for id,name in ipairs(species) do
  plankton_ids[name] = id
end

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(species)

myNet = convNet(opt)
-- this is the definition of the net. we will rewrite the command line arguments to be able to define this and some options at runtime.
-- then we can write a bash script to scan input space and find the best settings
-- myNet:build({1,64,128,64,#species}, 2, 2)

if opt.loadNet~='' then
  print( 'Loading '.. opt.netDatadir..'/'..opt.loadNet..'.dat')
  myNet.net = torch.load(opt.netDatadir..'/'..opt.loadNet..'.dat')
  myNet:reset()
else
  print('Building a new net to train')
  myNet:build({32,64}, {3,2,2}, {1,1,1}, {2,2,1})
end

dofile 'config_optimizer.lua'  -- optim.sgd, optim.asgd, optim.lbfgs, optim.cg 
dofile 'train.lua'  -- train convnet
dofile 'cross_validate.lua'  -- test convnet with cross-validation data
dofile 'save_nets.lua'

netSaver = save_nets(opt)
netSaver:prepNNdirs()

local points = {}
local time_stamps = {}
local scores = {}
--table.insert(scores,1e9)
for epoch = 1, opt.maxEpoch do
  train(epoch,myNet)
  local points_distr = test(epoch,myNet)
  local score = torch.sum(points_distr)
  if epoch ~= 1 and score>scores[epoch-1] then
    print('Score went up! Reducing learning rate')
    optimState.learningRate = optimState.learningRate * opt.learningRateScale
  end
  netSaver:saveNN(epoch,myNet:getNet())
  table.insert(scores,score)
  table.insert(points,points_distr)
  table.insert(time_stamps,os.date())
end
-- copy net with minimal cross-validation score to NN.dat
local min_val = torch.Tensor()
local min_index = torch.LongTensor()
torch.min(min_val,min_index,torch.Tensor(scores),1)
netSaver:saveBestNet(min_index[1],time_stamps[ min_index[1] ],scores[ min_index[1] ], points[ min_index[1] ], opt)
