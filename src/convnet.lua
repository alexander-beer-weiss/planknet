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

function convNet:getNet()
  return self.net
end
  

-- in Lua functions are first class citizens
local trans_layer ={}
trans_layer['Tanh']=nn.Tanh
trans_layer['ReLU']=nn.ReLU
trans_layer['Sigmoid']=nn.Sigmoid

-- Example build
-- parameters,gradParameters = myNet:build({1,64,64,128,#species}, 2, 5)
function convNet:build(dimensions, kW, dW, pools)
  self.net=nn.Sequential()
  local normkernel = image.gaussian1D(3)
  out_dimensions=#species
  print('These dimensions should be integers. If not then you need to add padding')
--   input to first layer
  self.net:add(nn.SpatialConvolutionMM(1, dimensions[1], kW[1], kW[1], dW[1], dW[1]))
  local owidth = (self.width - kW[1])/dW[1] + 1
  self.net:add(trans_layer[self.transfer]())
  self.net:add(nn.SpatialLPPooling(dimensions[1],2,pools[1],pools[1],pools[1],pools[1]))
  owidth = owidth/pools[1]
  self.net:add(nn.SpatialSubtractiveNormalization(dimensions[1], normkernel))
  self.net:add(nn.Dropout(self.dropout[1])) 
  print (owidth..' layer 1 output: width and height')
  print (owidth*owidth*dimensions[1]..' number of features')

  --   first to second layer
  self.net:add(nn.SpatialConvolutionMM(dimensions[1], dimensions[2], kW[2], kW[2], dW[2], dW[2]))
  owidth = (owidth - kW[2])/dW[2] + 1
  self.net:add(trans_layer[self.transfer]())
  self.net:add(nn.SpatialLPPooling(dimensions[2],2,pools[2],pools[2],pools[2],pools[2]))
  owidth = owidth/pools[2]
  self.net:add(nn.SpatialSubtractiveNormalization(dimensions[2], normkernel))
  self.net:add(nn.Dropout(self.dropout[2])) 
  print (owidth..' layer 2 output: width and height')
  print (owidth*owidth*dimensions[2]..' layer 2: number of features')
  local  linearFeatDim = owidth*owidth*dimensions[2]
  
-- --   second layer to linear
  if #dimensions==3 then
    self.net:add(nn.SpatialConvolutionMM(dimensions[2], dimensions[3], kW[3], kW[3], dW[3], dW[3]))
    owidth = (owidth - kW[3])/dW[3] + 1
    self.net:add(trans_layer[self.transfer]())
    self.net:add(nn.SpatialLPPooling(dimensions[3],2,pools[3],pools[3],pools[3],pools[3]))
    owidth = owidth/pools[3]
    self.net:add(nn.SpatialSubtractiveNormalization(dimensions[3], normkernel))
    self.net:add(nn.Dropout(self.dropout[3])) 
    print (owidth..' layer 3 output: width and height')
    print (owidth*owidth*dimensions[3]..' layer 3: number of features')
    linearFeatDim = owidth*owidth*dimensions[3]
    print(linearFeatDim..' final out layer')
    self.net:add(nn.Reshape(linearFeatDim))
    self.net:add(nn.Dropout(self.dropout[4]))
  else
    print(linearFeatDim..' final out layer')
    self.net:add(nn.Reshape(linearFeatDim))
    self.net:add(nn.Dropout(self.dropout[3]))
  end

--   linear to final
  self.net:add(nn.Linear(linearFeatDim, linearFeatDim))
  self.net:add(trans_layer[self.transfer]())
  self.net:add(nn.Linear(linearFeatDim, out_dimensions))

--   Final Layer
  self.net:add(nn.LogSoftMax())
  
  self.criterion = nn.ClassNLLCriterion()
  self.parameters, self.gradParameters = self.net:getParameters()
end

function convNet:reset()
  self.parameters, self.gradParameters = self.net:getParameters()
end


function convNet:n_batch()
  return 8
end

function convNet:trainStep(img,label)
  local output=self.net:forward(img)
  local err = self.criterion:forward(output,label)
  local df_dw = self.criterion:backward(output,label)
  self.net:backward(img,df_dw)
  return {err=err, output=output}
end

function convNet:augmentedTrainStep(img,label)
  local val = self:trainStep(img,label)
  local err = val['err']
  local output = {}
  table.insert(output, val['output'])
  val = self:trainStep(image.hflip(img),label)
  err = err + val['err']
  table.insert(output, val['output'])
  local tmp_img = img
  for i=1,3 do
    tmp_img = image.rotate(tmp_img,math.pi*0.5)
    val = self:trainStep(tmp_img,label)
    err = err + val['err']
    table.insert(output, val['output'])
    val = self:trainStep(image.hflip(tmp_img),label) 
    err = err + val['err']
    table.insert(output, val['output'])
  end
  return err, output
end
