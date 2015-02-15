-- takes a list of arrays and reorders indices of arrays
-- assumes all tables in ... are the same length; all arrays are shuffled into same order.
-- example: takes array of plankton images and array of associated labels and shuffles such that labels still match images
function table_shuffle(...)
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


function train(epoch,localNet)  -- epoch counts number of times through training data

	-- local vars
	local time = sys.clock()

	-- do one epoch
	print('==> doing epoch on training data:')
	print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	
	plankton_images_train, plankton_targets_train, plankton_paths_train
	   = table_shuffle(plankton_images_train,plankton_targets_train, plankton_paths_train)

	local misclassify = {}

	for batch_start_example = 1,#plankton_targets_train,opt.batchSize do
		-- disp progress
		xlua.progress(batch_start_example, #plankton_targets_train)

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
	                 for batch_example = batch_start_example,math.min(batch_start_example + opt.batchSize - 1, #plankton_targets_train) do
					    batch_size = batch_size + 1
						
	                    local output = localNet:forward(plankton_images_train[batch_example])
						
						--[[
						local max_val = torch.DoubleTensor()
						local max_index = torch.LongTensor()
						output.max(max_val,max_index,output,1)
						--print('Prediction: ' .. species[ max_index[1] ])
						if species[ max_index[1] ] ~= plankton_targets_train[batch_example] then
							misclassify[ plankton_targets_train[batch_example] ] = misclassify[ plankton_targets_train[batch_example] ] or {}
							misclassify[ plankton_targets_train[batch_example] ][ species[ max_index[1] ] ]
								= misclassify[ plankton_targets_train[batch_example] ][ species[ max_index[1] ] ] or {}
							table.insert(misclassify[ plankton_targets_train[batch_example] ][ species[ max_index[1] ] ],
							              plankton_paths_train[ batch_example ] )
						end
						--]]
						
	                    local err = criterion:forward(output, plankton_ids[ plankton_targets_train[batch_example] ])
	                    f = f + err
						
	                    -- estimate df/dW
	                    local df_do = criterion:backward(output, plankton_ids[ plankton_targets_train[batch_example] ])
	                    localNet:backward(plankton_images_train[batch_example], df_do)
						
	                    -- update confusion
	                    confusion:add(output, plankton_ids[ plankton_targets_train[batch_example] ])
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
	-- 	time = sys.clock() - time
	-- 	time = time / #plankton_images_train
	--         print("==> time to learn 1 sample = " .. (time*1000) .. 'ms')

	-- print confusion matrix
	-- print(confusion)
	confusion:zero()

	-- update logger/plot
	--trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
	--if opt.plot then
	--	trainLogger:style{['% mean class accuracy (train set)'] = '-'}
	--	trainLogger:plot()
	--end

	-- save/log current net
	--local filename = paths.concat(opt.save, 'localNet.net')
	--os.execute('mkdir -p ' .. sys.dirname(filename))
	--print('==> saving localNet to '..filename)
	--torch.save(filename, localNet)

	
	confusion:zero()

end
