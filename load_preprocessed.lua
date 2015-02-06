
print '==> loading preprocessed features'


-- loads tables:  image_targets, image_paths, species, species_count
dofile(preprocessed_path..'/plankton_tables.lua')


local preprocessed_file = hdf5.open(preprocessed_path..'/preprocessed_data.h5', 'r')
image_files = nn.SplitTable(1):forward(preprocessed_file:read('datapath'):all())
preprocessed_file:close()

