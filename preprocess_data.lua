
--------------------------------------------------

print '==> padding images'

plankton_images, plankton_labels, plankton_files  = table_shuffle(plankton_images, plankton_labels, plankton_files)

-- scan through images and find the maximum dimensions; only consider first 100 images
height, width = maxdims(plankton_images,100)

-- we may need to accommadate larger images than what we've scanned through
-- (images still larger than this will be cropped via pad function)
height, width = math.ceil(height * 1.2), math.ceil(width * 1.2)
                                                                       
height, width = height + height % 4, width + width % 4  -- accommodate pooling (two layers of 2x2 pooling)
-- print('height: ' .. height .. '  width: ' .. width)

height,width = 128,128  -- Hack!!!  Program still crashes sometimes.  128x128 is probably a bit too small


-- make all images the same dimensions
pad(plankton_images, height, width, true)  -- true inverts pixels (white <-> black) so that matrices become sparse







--------------------------------------------------

print '==> more preprocessing'





