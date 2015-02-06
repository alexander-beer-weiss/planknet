
if not paths.dir(preprocessed_path) then
	print('==> creating directory '..preprocessed_path)
	paths.mkdir(preprocessed_path)
end


-- NOW SAVE DATA TO HDF5 format

--plankton_images --> ../preprocessed/images.h5        -- lua array with values of type torch.DoubleTensor(number,number,number)
--plankton_labels --> ../preprocessed/targets.h5       -- lua array with values of type string
--plankton_files --> ../preprocessed/paths.h5          -- lua array with values of type string
--species --> ../preprocessed/species.h5               -- lua array with values of type string

