function goplankton()
	print '==> train and test'
	
	local epoch = 0
	
	if not paths.dir(misclassdata_path) then
		print('==> creating directory '..misclassdata_path)
		paths.mkdir(misclassdata_path)
	end
	local misclass_stream = io.open(misclassdata_path..'/misclassdata.lua','w')
	misclass_stream:write('images_path="',images_path,'"\n')
	misclass_stream:write('maxEpoch=',tostring(opt.maxEpoch),'\n')
	misclass_stream:write('height,width=',tostring(height),',',tostring(width),'\n')
	misclass_stream:write('misclassified={}\n')
	misclass_stream:close()

	local scan = true
	while epoch ~= opt.maxEpoch do
		epoch = epoch + 1		
   	 	train(epoch)
   		test(epoch)
	end
end
