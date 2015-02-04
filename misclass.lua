require 'torch'
require 'nn'
require 'image'

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
	end
end

dofile '../misclassified/misclassdata.lua'
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