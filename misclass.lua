require 'torch'
require 'nn'
require 'image'

planknet_path = '../planknet'

dofile(planknet_path..'/helper_functions.lua')
dofile 'misclass_data.lua'

print('Epoch: '..maxEpoch..':')
print(' ')
for specs_target,targets_table in pairs(misclassified[maxEpoch]) do
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
			misclass_tensor = nn.JoinTable(1):forward{torch.reshape(padded_image_file,1,1,height,width),misclass_tensor}
		end
		--print('IMAGE DIM: ' .. misclass_tensor:size(2) .. ',' .. misclass_tensor:size(3))
		itorch.image(misclass_tensor)
	end
end 