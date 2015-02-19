

-- test function
function test(predictions, netObject, test_batch, batch_num)
        local localNet=netObject:getNet()
	-- local vars
	local time = sys.clock()

	-- test over test data
	print('==> testing batch ' .. batch_num)
	misclassify = {}
	for img_index = 1, #test_batch do
		-- disp progress
		xlua.progress(img_index, #test_batch)

		-- test_batch is an array with values of the form { [1] = file_name, [2] = image }
		predictions[ test_batch[img_index][1] ] = torch.exp( localNet:forward(test_batch[img_index][2]) )
	
	end

	-- timing
	time = sys.clock() - time
	time = time / #test_batch
	print("==> time to test 1 sample = " .. (time*1000) .. 'ms')

end

