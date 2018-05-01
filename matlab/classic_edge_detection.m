% Example usage: classic_edge_detection(img_list, 'C:\DAN_PC\Facultate\Master\Deep Learning\Dataset\small_sample\bw', 'Sobel', '\')

function  classic_edge_detection(img_path_list, destination_path, der_type, path_separator)
% Returns all image paths at the specified based directory path
% 
% INPUT
% img_path_list - a list of strings, each representing a path to an image
% destination_path - destination path, where the edge detected images are
% stored
% der_type - the type of derivative to be used in edge detection
% path_separator - a string, used as separator in system paths

for idx = 1 : length(img_path_list)
    % Process the image
    image = imread(char(img_path_list(idx)));
    processed_img = edge_detect(image, der_type);
    
    % Store the image
    filename = strsplit(img_path_list(idx), path_separator);
    filename = strcat("edge_", filename(end));
    store_path = fullfile(char(destination_path), char(filename));
    imwrite(processed_img, store_path)
end
    
end

function processed_img = edge_detect(image, der_type)
% Perform some gamma correction
image = imadjust(image, [], [], 0.5);

% Smooth with a Gaussian
sigma = 2;
image = imgaussfilt(image, sigma);

% We'll grayscale the image
image_gray = 0.2989 * image(:, :, 1) + 0.5870 * image(:, :, 2) + 0.1140 * image(:, :, 3);
image = double(image_gray);

classType = class(image_gray);                 
scalingFactor = double(intmax(classType));  
image = image / scalingFactor;

% Finally, perform the edge detection 
processed_img = edge(image, der_type);
end


