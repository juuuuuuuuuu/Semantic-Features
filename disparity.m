for i = 1:271
    str = string(i-1)
    
    if (i-1) < 10
        cd '/Users/berkn/Desktop/ETH/Master/Semester_2/3D Vision/dataset/sequences/04/image_2'
        I1 = imread(strcat('00000', str, '.png'));
        cd '/Users/berkn/Desktop/ETH/Master/Semester_2/3D Vision/dataset/sequences/04/image_3'
        I2 = imread(strcat('00000', str, '.png'));
    end
    
    if (i-1) > 9 && (i-1) < 99
        cd '/Users/berkn/Desktop/ETH/Master/Semester_2/3D Vision/dataset/sequences/04/image_2'
        I1 = imread(strcat('0000', str, '.png'));
        cd '/Users/berkn/Desktop/ETH/Master/Semester_2/3D Vision/dataset/sequences/04/image_3'
        I2 = imread(strcat('0000', str, '.png'));
    end
    
    if (i-1) > 100
        cd '/Users/berkn/Desktop/ETH/Master/Semester_2/3D Vision/dataset/sequences/04/image_2'
        I1 = imread(strcat('000', str, '.png'));
        cd '/Users/berkn/Desktop/ETH/Master/Semester_2/3D Vision/dataset/sequences/04/image_3'
        I2 = imread(strcat('000', str, '.png'));
    end
    

    J1 = rgb2gray(I1);
    J2 = rgb2gray(I2);
    disparityRange = [0 480*2];
    disparityMap = disparityBM(J1,J2,'DisparityRange',disparityRange,'UniquenessThreshold',20);
    
    %disparityM = ind2gray(disparityMap, gray);

    %figure
    %imshow(disparityMap,disparityRange)
    %title('Disparity Map')
    %colormap gray
    %colorbar
    
    %handle = image(disparityMap);
    %imgmodel = imagemodel(handle);
   
    cd '/Users/berkn/Desktop/ETH/Master/Semester_2/3D Vision/dataset/sequences/04/disparity_new_range'
    imwrite(disparityMap, gray, strcat('0', str,'.png'));
    [X, map] = imread(strcat('0', str,'.png'));
    %O = ind2rgb(X,map);
    %O = rgb2gray(O);
    O = ind2gray(X,map);
    cd '/Users/berkn/Desktop/ETH/Master/Semester_2/3D Vision/dataset/sequences/04/disparity'
    imwrite(O, strcat(str,'.png'), 'BitDepth', 16);
    %[Y, map2] = imread(strcat(str,'.png'));
    
end
