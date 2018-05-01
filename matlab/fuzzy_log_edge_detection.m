% Example usage: fuzzy_log_edge_detection(img_list, 'C:\DAN_PC\Facultate\Master\Deep Learning\Dataset\small_sample\bw', '\', 0.2, 0.2, 0.98)

function  fuzzy_log_edge_detection(img_path_list, destination_path, path_separator, sx, sy, fraction)
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
    processed_img = FLED(image, sx, sy, fraction);
    
    % Store the image
    filename = strsplit(img_path_list(idx), path_separator);
    filename = strcat("edge_", filename(end));
    store_path = fullfile(char(destination_path), char(filename));
    imwrite(processed_img, store_path)
end
    
end

function Ieval = FLED(Irgb, sx, sy, fraction)
% Note: higher values for sx and sy make edge detection less sensitive


% Perform some gamma correction
Irgb = imadjust(Irgb, [], [], 0.5);

% Smooth with a Gaussian
sigma = 2;
Irgb = imgaussfilt(Irgb, sigma);

% Grayscale the image
Igray = 0.2989*Irgb(:,:,1)+0.5870*Irgb(:,:,2)+0.1140*Irgb(:,:,3);

% figure
% image(Igray,'CDataMapping','scaled');
% colormap('gray')
% title('Input Image in Grayscale');

I = double(Igray);

classType = class(Igray);                 
scalingFactor = double(intmax(classType));  
I = I/scalingFactor;

% Compute the horizontal / vertical gradients
Gx = [-1 1];
Gy = Gx';
Ix = conv2(I,Gx,'same');
Iy = conv2(I,Gy,'same');

% figure
% image(Ix,'CDataMapping','scaled')
% colormap('gray')
% title('Ix')

% figure
% image(Iy,'CDataMapping','scaled')
% colormap('gray')
% title('Iy')

% Create a fuzzy inference system, and add the gradients as variables
edgeFIS = newfis('edgeDetection');
edgeFIS = addvar(edgeFIS,'input','Ix',[-1 1]);
edgeFIS = addvar(edgeFIS,'input','Iy',[-1 1]);

% Add gaussmf as a membership function to the FIS
edgeFIS = addmf(edgeFIS,'input',1,'zero','gaussmf',[sx 0]);
edgeFIS = addmf(edgeFIS,'input',2,'zero','gaussmf',[sy 0]);

edgeFIS = addvar(edgeFIS,'output','Iout',[0 1]);

% Parameters for the triangular-shaped membership function 
wa = 0.1;
wb = 1;
wc = 1;
ba = 0;
bb = 0;
bc = 0.7;

% Add a triangular-shaped membership function to the FIS
edgeFIS = addmf(edgeFIS,'output',1,'white','trimf',[wa wb wc]);
edgeFIS = addmf(edgeFIS,'output',1,'black','trimf',[ba bb bc]);

r1 = 'If Ix is zero and Iy is zero then Iout is white';
r2 = 'If Ix is not zero or Iy is not zero then Iout is black';
r = char(r1,r2);
edgeFIS = parsrule(edgeFIS,r);
% showrule(edgeFIS)

Ieval = zeros(size(I));
for ii = 1:size(I,1)
    Ieval(ii,:) = evalfis([(Ix(ii,:));(Iy(ii,:));]',edgeFIS);
end

avg_val = mean(mean(Ieval)) * fraction;

disp(avg_val);

Ieval = imbinarize(Ieval, avg_val);

% figure
% image(Ieval,'CDataMapping','scaled')
% colormap('gray')
% title('Edge Detection Using Fuzzy Logic')

end