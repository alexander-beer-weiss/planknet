print '==> defining some tools'

-- classes
plankton_ids={}
for id,name in ipairs(species) do
	plankton_ids[name] = id
end

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(species)

-- Log results to files
--trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
--testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode into a 1-dim vector
parameters,gradParameters = convnet:getParameters()
