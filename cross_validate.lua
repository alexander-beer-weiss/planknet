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
	for test_example = 1,#plankton_targets_cv do
		-- disp progress
		xlua.progress(test_example, #plankton_targets_cv)

		-- test sample
		local pred = convnet:forward(plankton_images_cv[test_example])

		local max_val = torch.Tensor()
		local max_index = torch.LongTensor()
		pred.max(max_val,max_index,pred,1)
		--print('Prediction: ' .. species[ max_index[1] ])
		if species[ max_index[1] ] ~= plankton_targets_cv[test_example] then
			misclassify[ plankton_targets_cv[test_example] ] = misclassify[ plankton_targets_cv[test_example] ] or {}
			misclassify[ plankton_targets_cv[test_example] ][ species[ max_index[1] ] ]
				= misclassify[ plankton_targets_cv[test_example] ][ species[ max_index[1] ] ] or {}
			table.insert(misclassify[ plankton_targets_cv[test_example] ][ species[ max_index[1] ] ],
			              plankton_paths_cv[ test_example ] )
		end



		confusion:add(pred, plankton_ids[ plankton_targets_cv[test_example] ])
	end

	-- timing
	time = sys.clock() - time
	time = time / #plankton_targets_cv
	print("==> time to test 1 sample = " .. (time*1000) .. 'ms')

	-- print confusion matrix
	print(confusion)
	confusion:zero()

	-- print misclassifed cv examples to file
	local misclass_stream = io.open(misclassdata_path..'/misclass_data.lua','a')
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
