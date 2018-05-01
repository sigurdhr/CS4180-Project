% Example usage: img_list = get_all_images('C:\DAN_PC\Facultate\Master\Deep Learning\Dataset\small_sample')

function image_path_list = get_all_images(path)
% Returns all image paths at the specified based directory path
% 
% INPUT
% path - path to the base directory containing a number of images to be
% processed; the images must have a valid type (.jpeg, .png, etc.), else,
% they won't be considered
% path_separator - a string, used as separator in system paths
% 
% OUTPUT
% image_path_list - a list of fully qualified (absolute) paths to the
% images at the path folder

% Hardcode a list of valid file formats
valid_formats = [...
    '.BMP' '.GIF' '.HDF' '.JPG' '.JPEG' ...
    '.JP2' '.JPX' '.PBM' '.PCX' '.PGM' ...
    '.PNG' '.PNM''.PPM' '.RAS' '.TIF' ...
    '.TIFF' '.XWD'
    ]; 

% Get the files in the dir then iterate over them
file_listing = dir(path);

% Pre-allocate for speed
image_path_list = strings(length(file_listing), 1);
counter = 0;

for file_idx = 1 : length(file_listing)
    [~, ~, ext] = fileparts(file_listing(file_idx).name);
    
    if ismember(ext, valid_formats) & ext ~= "."
        counter = counter + 1;
        image_path_list(counter) = fullfile(file_listing(file_idx).folder, file_listing(file_idx).name);
    end
end

% Remove the unused array elements
image_path_list = image_path_list(1:counter, 1);

end