% Calculates disparitymaps for folder
% /Users/johannesconradi/Downloads/dataset/sequences/04/image_...
%@Felix Change directory to left and right pic

filenamestam = 'raw.png';
disparityRange = [0 48];

myDir_I1 = '/Users/johannesconradi/Downloads/kitti_dataset/dataset/sequences/04/image_0';
myDir_I2 = '/Users/johannesconradi/Downloads/kitti_dataset/dataset/sequences/04/image_1'; 
myFiles_I1 = dir(fullfile(myDir_I1,'*.png'));
myFiles_I2 = dir(fullfile(myDir_I2,'*.png'));
for k = 1:length(myFiles_I2)
    baseFileName_I1 = myFiles_I1(k).name;
    baseFileName_I2 = myFiles_I2(k).name;
    fullFileName_I1 = fullfile(myDir_I1, baseFileName_I1);
    fullFileName_I2 = fullfile(myDir_I2, baseFileName_I2);
    fprintf(1, 'Now reading %s\n', fullFileName_I2);
    
    I1 = imread(fullFileName_I1);
    I2 = imread(fullFileName_I2);
    
    
    disparityMap = disparitySGM(I1,I2,'DisparityRange',disparityRange,'UniquenessThreshold',20);
    disparityMap = disparityMap/48;
    filename = strcat(int2str(k-1),filenamestam);
    imwrite(disparityMap,filename, 'BitDepth',16)
end