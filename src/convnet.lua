require 'nn'
require 'image'
require 'underscore'

convNet = {}
convNet.__index = convNet
setmetatable(convNet, {
  __call = function (cls, ...)
    return cls.new(...)
  end,
})

function convNet.new(opt)
  local self = setmetatable({},convNet)
  for key, value in pairs(opt) do
    self[key] = value
  end
  return self
end


-- in Lua functions are first class citizens
local trans_layer ={}
trans_layer['Tanh']=nn.Tanh
trans_layer['ReLU']=nn.ReLU
trans_layer['Sigmoid']=nn.Sigmoid

-- Example build
-- parameters,gradParameters = myNet:build({1,64,64,128,#species}, 2, 5)
function convNet:build(dimensions, n_pools, n_filter)
  self.net=nn.Sequential()
  local normkernel = image.gaussian1D(3)
  n_dimensions=#dimensions
  for i=1,n_dimensions-3 do
    if self.dropout[n_dimensions - i] and self.dropout[n_dimensions - i] ~= 0 then self.net:add(nn.Dropout(self.dropout[n_dimensions - i])) end
    self.net:add(nn.SpatialConvolutionMM(dimensions[i], dimensions[i+1], n_filter, n_filter))
    self.net:add(trans_layer[self.transfer]())
    self.net:add(nn.SpatialLPPooling(dimensions[i+1],2,n_pools,n_pools,n_pools,n_pools))
    self.net:add(nn.SpatialSubtractiveNormalization(dimensions[i+1], normkernel))
  end
--   N-2 layer
  local linearFeatDim = dimensions[n_dimensions-2] * (((self.height-n_filter+1)/n_pools - n_filter + 1 )/n_pools)  * (((width-n_filter+1)/n_pools - n_filter + 1 )/n_pools)
--   print('feature dim: ' .. linearFeatDim )
  self.net:add(nn.Reshape(linearFeatDim))
  if self.dropout[2] and self.dropout[2] ~= 0 then self.net:add(nn.Dropout(self.dropout[1])) end
  
--   N-1 Layer
  self.net:add(nn.Linear(linearFeatDim, dimensions[n_dimensions-1]))
  self.net:add(trans_layer[self.transfer]())
  if self.dropout[1] and self.dropout[1] ~= 0 then self.net:add(nn.Dropout(self.dropout[1])) end
  self.net:add(nn.Linear(dimensions[n_dimensions-1], dimensions[n_dimensions]))

--   Final Layer
  self.net:add(nn.LogSoftMax())
  
  self.criterion = nn.ClassNLLCriterion()
  self.parameters, self.gradParameters = self.net:getParameters()
end

function convNet:n_batch()
  return 8
end

function convNet:augmentedForwardNet(img)
  local output={}
  output[1] = self.net:forward(img)
  output[2] = self.net:forward(image.hflip(img))
--   translate, invert, rotate
  local tmp_img = img
  for i=1,3 do
    tmp_img = image.rotate(tmp_img,math.pi*0.5)
    output[2*i+1] = self.net:forward(tmp_img)
    output[2*i+2] = self.net:forward(image.hflip(tmp_img))
  end
  return output
end

function convNet:augmentedForwardCriterion(output, label)
  local err = {}
  for i=1,8 do
    err[i] = self.criterion:forward(output[i],label)
  end
  return err
end

function convNet:augmentedBackwardsNet(img,df_dw)
  local output={}
  output[1] = self.net:backward(img,df_dw[1])
  output[2] = self.net:backward(image.hflip(img),df_dw[2])
--   translate, invert, rotate
  local tmp_img = img
  for i=1,3 do
    tmp_img = image.rotate(tmp_img,math.pi*0.5)
    output[2*i+1] = self.net:backward(tmp_img,df_dw[2*i+1])
    output[2*i+2] = self.net:backward(image.hflip(tmp_img),df_dw[2*i+2])
  end
  return output
end

function convNet:augmentedBackwardsCriterion(output, label)
  local err = {}
  for i=1,8 do
    err[i] = self.criterion:backward(output[i],label)
  end
  return err
end