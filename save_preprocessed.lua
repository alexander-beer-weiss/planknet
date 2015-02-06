
if not paths.dir(preprocessed_path) then
	print('==> creating directory '..preprocessed_path)
	paths.mkdir(preprocessed_path)
end



-- NOW SAVE DATA TO HDF5 format

--plankton_images --> ../preprocessed/images.h5        -- lua array with values of type torch.DoubleTensor(number,number,number)
--plankton_labels --> ../preprocessed/targets.h5       -- lua array with values of type string
--plankton_files --> ../preprocessed/paths.h5          -- lua array with values of type string
--species --> ../preprocessed/species.h5               -- lua array with values of type string



print '==> saving preprocessed data to disk'


local image_Tensor = torch.Tensor(#plankton_images,1,height,width)


local time = sys.clock()
for i = 1,#plankton_images do
	--for j = 1,height do
		--for k = 1,width do
			image_Tensor[i] = plankton_images[i]:clone()
			--end
	--end
end


local preprocessed_file = hdf5.open(preprocessed_path..'/preprocessed_data.h5', 'w')
preprocessed_file:write('datapath', image_Tensor)
preprocessed_file:close()


plankton_tables = {['image_targets'] = plankton_labels, ['image_paths'] = plankton_files,
                               ['species'] = species, ['species_count'] = species_count}

local file_stream = io.open(preprocessed_path..'/plankton_tables.lua','w')
for table_name,table_data in pairs(plankton_tables) do
	file_stream:write(table_name,'=')
	file_stream:write('{')
	for _,i in ipairs(table_data) do
		file_stream:write('"',i,'",')
	end
	file_stream:write('}\n')
end
file_stream:close()

