% this script builds a dataset of OCR from a print font Ubuntu
% things to do:
% normalize the size of the sample to 50 * 50
% center the contents (may be the hardest part! )
% rotate the content left and right with angle 15 and 30
% add pepper noise to the image

filenames = dir('dataset');
label = [];
data = [];
for i = 3:length(filenames)
    img = im2double(imread(strcat('dataset/',filenames(i).name)));
    img = double(~im2bw(img, 0.5));
    img = imresize(img, [25, 25]);
    img = imresize(img, [50, 50]);
    imgn = imnoise(img, 'salt & pepper');
    data = [data, img(:)];
    label = [label, i-3];
    data = [data, imgn(:)];
    label = [label, i-3];
    
    % rotate the image and add noise
    imgt = imresize(imrotate(img, 10), [50,50]);
    data = [data, imgt(:)];
    label = [label, i-3];
    imgt = imnoise(imgt, 'salt & pepper');
    data = [data, imgt(:)];
    label = [label, i-3];
    
    imgt = imresize(imrotate(img, 15), [50,50]);
    data = [data, imgt(:)];
    label = [label, i-3];
    imgt = imnoise(imgt, 'salt & pepper');
    data = [data, imgt(:)];
    label = [label, i-3];
    
    imgt = imresize(imrotate(img, 20), [50,50]);
    data = [data, imgt(:)];
    label = [label, i-3];
    imgt = imnoise(imgt, 'salt & pepper');
    data = [data, imgt(:)];
    label = [label, i-3];
    
    imgt = imresize(imrotate(img, 25), [50,50]);
    data = [data, imgt(:)];
    label = [label, i-3];
    imgt = imnoise(imgt, 'salt & pepper');
    data = [data, imgt(:)];
    label = [label, i-3];
    
    imgt = imresize(imrotate(img, 30), [50,50]);
    data = [data, imgt(:)];
    label = [label, i-3];
    imgt = imnoise(imgt, 'salt & pepper');
    data = [data, imgt(:)];
    label = [label, i-3];
    
    imgt = imresize(imrotate(img, 350), [50,50]);
    data = [data, imgt(:)];
    label = [label, i-3];
    imgt = imnoise(imgt, 'salt & pepper');
    data = [data, imgt(:)];
    label = [label, i-3];
    
    imgt = imresize(imrotate(img, 340), [50,50]);
    data = [data, imgt(:)];
    label = [label, i-3];
    imgt = imnoise(imgt, 'salt & pepper');
    data = [data, imgt(:)];
    label = [label, i-3];
    
    imgt = imresize(imrotate(img, 330), [50,50]);
    data = [data, imgt(:)];
    label = [label, i-3];
    imgt = imnoise(imgt, 'salt & pepper');
    data = [data, imgt(:)];
    label = [label, i-3];
end

save('built_new.mat', 'data', 'label');
