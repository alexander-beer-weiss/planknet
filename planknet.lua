
require 'torch'
require 'nn'
require 'xlua'
require 'optim'
require 'image'
require 'paths'


-- CmdLine() doesn't work in iTorch notebook.
-- Note:  Only SGD is currently fully supported.


----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Training/Optimization')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-visualize', false, 'visualize input data and weights during training')
   cmd:option('-plot', false, 'live plot')
   cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
   cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
   cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
   cmd:option('-momentum', 0, 'momentum (SGD only)')
   cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
   cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
   cmd:option('-maxEpoch', -1, 'maximum number of epochs during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

--[[
opt = {}
opt.save = 'results'
opt.visualize= false
opt.plot = false
opt.optimization = 'SGD'
opt.learningRate = 1e-3
opt.batchSize = 1
opt.weightDecay = 0
opt.momentum = 0
opt.t0 = 1
opt.maxIter = 2
opt.maxEpoch = 2
--]]


----------------------------------------------------------------------
print '==> import images'

images_path = '../images'
misclassdata_path = '../misclassified'

-- first some helper functions

-- takes an array of images of type torch.Tensor(1, num_rows, num_cols) -- the 1 means one channel; i.e., greyscale image
-- returns the maximum height and maximum width from the full set of images
-- optional parameter max_scan defines the maximum number of images to scan through before returning max dimensions
local function maxdims(image_array, max_scan)
	local maxheight, maxwidth = 0, 0
	for i = 1,#image_array do
		if maxheight < image_array[i]:size(2) then
			maxheight = image_array[i]:size(2)
		end
		if maxwidth < image_array[i]:size(3) then
			maxwidth = image_array[i]:size(3)
		end
		if max_scan then
			max_scan = max_scan - 1
			if max_scan == 0 then
				return maxheight, maxwidth
			end
		end
	end
	return maxheight, maxwidth
end

-- takes an array of images of type torch.Tensor(1, num_rows, num_cols), where num_rows and num_cols vary from image to image
-- returns an array of images of type torch.Tensor(1, height, width), where height and width are same for all images
-- optional parameter invert: if true inverts matrices (i.e., components go to 1-components) yielding a sparse matrix
local function pad(image_array, height, width, invert)
	if invert == true then
		padding = 0
	else
		padding = 1
	end


	for i = 1, #image_array do
		local padded_image = torch.Tensor(1,height,width):fill(padding)
		-- center image on padded canvas (probably not necessary for convolutional net but images display nicer in iTorch)
		local canvas_height_offset, canvas_width_offset, image_height_offset, image_width_offset = 0,0,0,0
		if image_array[i]:size(2) < height then
			canvas_height_offset = math.floor( (height - image_array[i]:size(2)) / 2 )
		else
			image_height_offset = math.floor( (image_array[i]:size(2) - height) / 2 )
		end
		if image_array[i]:size(3) < width then
			canvas_width_offset = math.floor( (width - image_array[i]:size(3)) / 2 )
		else
			image_width_offset = math.floor( (image_array[i]:size(3) - width) / 2 )
		end
		for row = 1, math.min(image_array[i]:size(2), height) do  -- oversized images are simply cropped
			for col = 1, math.min(image_array[i]:size(3), width) do
				if invert == true then
					--print(row + canvas_height_offset,col + canvas_width_offset,row + image_height_offset,col + image_width_offset)
					padded_image[1][row + canvas_height_offset][col + canvas_width_offset]
					  = 1 - image_array[i][1][row + image_height_offset][col + image_width_offset]  -- the [1] is the greyscale label index
				else
					padded_image[1][row + canvas_height_offset][col + canvas_width_offset]
					  = image_array[i][1][row + image_height_offset][col + image_width_offset]
				end
			end
		end
		image_array[i] = padded_image  -- overwrites original (unpadded) image
		--[[
		for i = 1, #image_array do
			local padded_image = torch.Tensor(1,height,width):fill(padding)
			for row = 1, math.min(image_array[i]:size(2), height) do  -- oversized images are simply cropped
				for col = 1, math.min(image_array[i]:size(3), width) do
					if invert == true then
						padded_image[1][row][col] = 1 - image_array[i][1][row][col]  -- the [1] is the greyscale label index
					else
						padded_image[1][row][col] = image_array[i][1][row][col]
				end
			end
		end
		image_array[i] = padded_image  -- overwrites original (unpadded) image
		--]]
	end
end


-- takes a list of arrays and reorders indices of arrays
-- assumes all tables in ... are the same length; all arrays are shuffled into same order.
-- example: takes array of plankton images and array of associated labels and shuffles such that labels still match images
local function table_shuffle(...)
	local table_of_tables = {...}
	local table_length = #table_of_tables[1]
	local shuffled_indices = torch.randperm(table_length)
	shuffled_table_of_tables = {}
	
	for i = 1,#table_of_tables do
		shuffled_table_of_tables[i] = {}
		for j = 1,table_length do
			table.insert(shuffled_table_of_tables[i], table_of_tables[i][ shuffled_indices[j] ])
		end
	end
	return unpack(shuffled_table_of_tables)
end


-- *** these should be local ***
plankton_images_train = {}  -- images of plankton ( array with values of type torch.DoubleTensor(num_channels, num_rows, num_cols) )
plankton_images_cv = {}  -- set aside 20% for cross-validation
plankton_labels_train = {}  -- labels for images  ( array with values of type string )
plankton_labels_cv = {}  -- set aside (same) 20% for cross-validation
plankton_files_train = {}  -- files, so we can visualize which ones our neural net misses
plankton_files_cv={}
local species = {}
for _,dir in pairs(paths.dir(images_path)) do  -- loop through image directories (extract names of sub-directories)
	if not string.match(dir,'^%.') then  -- ignore any directories (or files) starting with a period
		table.insert(species,dir)
		local image_files = {}
		for file in paths.files(images_path..'/'..dir) do  -- read in names of all jpeg files; make table of these names
			if string.match(file,'%.jpg$') then
				table.insert(image_files,file)
			end	
		end
		image_files = table_shuffle(image_files)
		local num_images = #image_files
		for i = 1, num_images do
			if i <= num_images * 0.8 then  -- 80% of examples used for training
				table.insert(plankton_images_train,image.load(images_path..'/'..dir..'/'..image_files[i]))  -- image is a torch.Tensor(num_channels, height, width)
				table.insert(plankton_labels_train,dir)  -- the directory is the species name; these are the labels
				table.insert(plankton_files_train,image_files[i])
				else  -- 20% of examples used for cross-validation
				table.insert(plankton_images_cv,image.load(images_path..'/'..dir..'/'..image_files[i]))  -- image is a torch.Tensor(num_channels, height, width)
				table.insert(plankton_labels_cv,dir)  -- the directory is the species name; these are the labels
				table.insert(plankton_files_cv,image_files[i])
			end
		end
	end
end

plankton_images_train, plankton_labels_train, plankton_files_train
   = table_shuffle(plankton_images_train,plankton_labels_train, plankton_files_train)

local height, width = maxdims(plankton_images_train,100)  -- scans through images and finds the maximum dimensions of jpegs; only consider first 100 images
height, width = math.ceil(height * 1.2), math.ceil(width * 1.2)  -- we may need to accommadate larger images than what we scanned through above
                                                                       -- (images still larger than this will be cropped via pad function)
height, width = height + height % 4, width + width % 4  -- accommodate pooling (two layers of 2x2 pooling)
-- print('height: ' .. height .. '  width: ' .. width)

height,width = 128,128  -- Hack!!!  Program still crashes sometimes.  128x128 is probably a bit too small

pad(plankton_images_train, height, width, true)  -- makes all images the same dimensions (true inverts pixels so matrices become sparse)
pad(plankton_images_cv, height, width, true)






-- Define relevant convnet parameters

-- 2-class problem
local noutputs = #species

-- input dimensions
local nfeats = 1  -- greyscale
--local ninputs = nfeats*width*height  -- number of inputs per image

-- hidden units
nstates = {64,64,128}

local filtsize = 5

-- pooling will be 2 x 2 for both layers (could be done differently for different layers using a table)
local poolsize = 2





----------------------------------------------------------------------

print '==> building convnet structure'

-- Build convnet structure

-- *** convnet, criterion, etc... could be local; what makes sense? ***

convnet = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
convnet:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
convnet:add(nn.Tanh())
--convnet:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
convnet:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
convnet:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
convnet:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
convnet:add(nn.Tanh())
convnet:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))  -- 2 means L2-norm
convnet:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

-- stage 3 : standard 2-layer neural network
local linearFeatDim = nstates[2] * (((height-filtsize+1)/poolsize - filtsize + 1 )/poolsize)  * (((width-filtsize+1)/poolsize - filtsize + 1 )/poolsize)
--print('feature dim: ' .. LinearFeatDim)
convnet:add(nn.Reshape(linearFeatDim))  -- THIS IS WHY WE CHOOSE height = width = 128.  linearFeatDim formula doesn't always work... why??
convnet:add(nn.Linear(linearFeatDim, nstates[3]))
convnet:add(nn.Tanh())
convnet:add(nn.Linear(nstates[3], noutputs))
convnet:add(nn.LogSoftMax())


-- loss function
criterion = nn.ClassNLLCriterion()
--criterion = optim.ConfusionMatrix()

-- retrieve parameters and gradients
parameters,gradParameters = convnet:getParameters()







----------------------------------------------------------------------
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
if convnet then
   parameters,gradParameters = convnet:getParameters()
end


----------------------------------------------------------------------
print '==> configuring optimizer'  -- ONLY SGD FULLY SUPPORTED AS OF NOW

if opt.optimization == 'CG' then
   optimState = {
      maxIter = opt.maxIter
   }
   optimMethod = optim.cg

elseif opt.optimization == 'LBFGS' then
   optimState = {
      learningRate = opt.learningRate,
      maxIter = opt.maxIter,
      nCorrection = 10
   }
   optimMethod = optim.lbfgs

elseif opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd

elseif opt.optimization == 'ASGD' then
   optimState = {
      eta0 = opt.learningRate,
      t0 = trsize * opt.t0
   }
   optimMethod = optim.asgd

else
   error('unknown optimization method')
end






----------------------------------------------------------------------


print '==> defining training procedure'

function train(epoch)  -- epoch counts number of times through training data

	-- local vars
	local time = sys.clock()


	-- do one epoch
	print('==> doing epoch on training data:')
	print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

	local misclassify = {}

	for batch_start_example = 1,#plankton_labels_train,opt.batchSize do
		-- disp progress
		xlua.progress(batch_start_example, #plankton_labels_train)

		-- create closure to evaluate f(X) and df/dX
		local feval = function(x)
	                 -- get new parameters
	                 if x ~= parameters then
	                    parameters:copy(x)
	                 end
					 --print(parameters)
					 
	                 -- reset gradients
	                 gradParameters:zero()
					 
	                 -- f is the average of all criterions
	                 local f = 0
					 
	                 -- evaluate function for complete mini batch
				     local batch_size = 0  -- keeps track of actual batch size (since last batch may be smaller)
	                 for batch_example = batch_start_example,math.min(batch_start_example + opt.batchSize - 1, #plankton_labels_train) do
					    batch_size = batch_size + 1
						
	                    local output = convnet:forward(plankton_images_train[batch_example])
						
						--[[
						local max_val = torch.DoubleTensor()
						local max_index = torch.LongTensor()
						output.max(max_val,max_index,output,1)
						--print('Prediction: ' .. species[ max_index[1] ])
						if species[ max_index[1] ] ~= plankton_labels_train[batch_example] then
							misclassify[ plankton_labels_train[batch_example] ] = misclassify[ plankton_labels_train[batch_example] ] or {}
							misclassify[ plankton_labels_train[batch_example] ][ species[ max_index[1] ] ]
								= misclassify[ plankton_labels_train[batch_example] ][ species[ max_index[1] ] ] or {}
							table.insert(misclassify[ plankton_labels_train[batch_example] ][ species[ max_index[1] ] ],
							              plankton_files_train[ batch_example ] )
						end
						--]]
	                    local err = criterion:forward(output, plankton_ids[ plankton_labels_train[batch_example] ])

	                    f = f + err
						
	                    -- estimate df/dW
	                    local df_do = criterion:backward(output, plankton_ids[ plankton_labels_train[batch_example] ])
	                    convnet:backward(plankton_images_train[batch_example], df_do)
						
	                    -- update confusion
	                    confusion:add(output, plankton_ids[ plankton_labels_train[batch_example] ])
	                 end
					 
	                 -- normalize gradients and f(X)
	                 gradParameters:div(batch_size)
	                 f = f / batch_size
					 
	                 -- return f and df/dX
	                 return f,gradParameters
	              end

		-- optimize on current mini-batch
		if optimMethod == optim.asgd then
			_,_,average = optimMethod(feval, parameters, optimState)
		else
			optimMethod(feval, parameters, optimState)
		end
	end

	-- time taken
	time = sys.clock() - time
	time = time / #plankton_images_train
	print("==> time to learn 1 sample = " .. (time*1000) .. 'ms')

	-- print confusion matrix
	print(confusion)
	confusion:zero()

	-- update logger/plot
	--trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
	--if opt.plot then
	--	trainLogger:style{['% mean class accuracy (train set)'] = '-'}
	--	trainLogger:plot()
	--end

	-- save/log current net
	--local filename = paths.concat(opt.save, 'convnet.net')
	--os.execute('mkdir -p ' .. sys.dirname(filename))
	--print('==> saving convnet to '..filename)
	--torch.save(filename, convnet)

	plankton_images_train, plankton_labels_train, plankton_files_train
	   = table_shuffle(plankton_images_train,plankton_labels_train, plankton_files_train)
	
	confusion:zero()

end



----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test(epoch)
	
	-- local vars
	local time = sys.clock()

	-- averaged param use?
	if average then
		cachedparams = parameters:clone()
		parameters:copy(average)
	end

	-- test over test data
	print('==> testing on test set:')
	misclassify = {}
	for test_example = 1,#plankton_labels_cv do
		-- disp progress
		xlua.progress(test_example, #plankton_labels_cv)

		-- test sample
		local pred = convnet:forward(plankton_images_cv[test_example]:double())

		local max_val = torch.DoubleTensor()
		local max_index = torch.LongTensor()
		pred.max(max_val,max_index,pred,1)
		--print('Prediction: ' .. species[ max_index[1] ])
		if species[ max_index[1] ] ~= plankton_labels_cv[test_example] then
			misclassify[ plankton_labels_cv[test_example] ] = misclassify[ plankton_labels_cv[test_example] ] or {}
			misclassify[ plankton_labels_cv[test_example] ][ species[ max_index[1] ] ]
				= misclassify[ plankton_labels_cv[test_example] ][ species[ max_index[1] ] ] or {}
			table.insert(misclassify[ plankton_labels_cv[test_example] ][ species[ max_index[1] ] ],
			              plankton_files_cv[ test_example ] )
		end



		confusion:add(pred, plankton_ids[ plankton_labels_cv[test_example] ])
	end

	-- timing
	time = sys.clock() - time
	time = time / #plankton_labels_cv
	print("==> time to test 1 sample = " .. (time*1000) .. 'ms')

	-- print confusion matrix
	print(confusion)
	confusion:zero()


	local misclass_stream = io.open(misclassdata_path..'/misclassdata.lua','a')
	misclass_stream:write('table.insert(misclassified,')
	misclass_stream:write('{')
	for a,g in pairs(misclassify) do
	  misclass_stream:write('["',a,'"]={')
	  for b,h in pairs(g) do
	    misclass_stream:write('["',b,'"]={')
	    for _,i in ipairs(h) do
	      misclass_stream:write('"',i,'",')
	    end
	    misclass_stream:write('},')
	  end
	  misclass_stream:write('},')
	end
	misclass_stream:write('})\n')
	misclass_stream:close()


	--[[	
	if epoch == opt.maxEpoch then
		print('MISCLASSIFIED:')
		print(misclassify)
		print(' ')
		for specs_target,targets_table in pairs(misclassify) do
			--print ('Examples of true ' .. specs_target .. ': ')
			
			for specs_mislabel,mislabeled_table in pairs(targets_table) do
				print('Target: '.. specs_target, 'Label: ' .. specs_mislabel)
				local misclass_table = {}
				for _,image_file in ipairs(mislabeled_table) do
					table.insert(misclass_table,image.load(images_path .. '/' .. specs_target .. '/' .. image_file))
				end
				pad(misclass_table,height,width)
				local misclass_tensor = nil
				for _,padded_image_file in ipairs(misclass_table) do
					-- could alternatively torch.reshape(padded..., 1, 1, height, width) and then JoinTable(1)
					misclass_tensor = nn.JoinTable(3):forward{padded_image_file,misclass_tensor}
				end
				--print('IMAGE DIM: ' .. misclass_tensor:size(2) .. ',' .. misclass_tensor:size(3))
				itorch.image(misclass_tensor)
			end
		end 
	end
	--]]

	-- update log/plot
	--testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
	--if opt.plot then
	--	testLogger:style{['% mean class accuracy (test set)'] = '-'}
	--	testLogger:plot()
	--end

	-- averaged param use?
	if average then
		-- restore parameters
		parameters:copy(cachedparams)
	end
end


----------------------------------------------------------------------

function goplankton()
	print '==> train and test'
	
	local epoch = 0
	
	if not os.execute('ls '..misclassdata_path) then os.execute('mkdir '..misclassdata_path) end
	local misclass_stream = io.open(misclassdata_path..'/misclassdata.lua','w')
	misclass_stream:write('images_path="',images_path,'"\n')
	misclass_stream:write('maxEpoch=',tostring(opt.maxEpoch),'\n')
	misclass_stream:write('height,width=',tostring(height),',',tostring(width),'\n')
	misclass_stream:write('misclassified={}\n')
	misclass_stream:close()

	local scan = true
	while epoch ~= opt.maxEpoch do
		epoch = epoch + 1		
   	 	train(epoch)  -- scan = false on last epoch (should change this)
   		test(epoch)
	end
end


