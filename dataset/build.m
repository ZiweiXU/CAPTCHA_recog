% this script builds a dataset from ./sliced
% the final dataset will be wrote to built_from_mis.mat, which can be later used by train.py
% the first character of the filename of the image will be the label
filenames = dir('sliced');
label = [];
data = [];

for i = 3:length(filenames)
    img = im2double(imread(strcat('sliced/',filenames(i).name)));
    img = imresize(img, [50, 50]);
    this_label = filenames(i).name(1);
    if(this_label < '9')
        this_label = str2double(this_label);
    else
        this_label = 10 + double(this_label) - double('A');
    end
    
    img = im2double(imread(strcat('sliced/',filenames(i).name)));
    img = imresize(img, [50, 50]);
    imgn = imnoise(img, 'salt & pepper');
    data = [data, img(:)];
    label = [label, this_label];
    data = [data, imgn(:)];
    label = [label, this_label];
    
    % rotate the image and add noise
    imgt = imresize(imrotate(img, 10), [50,50]);
    data = [data, imgt(:)];
    label = [label, this_label];
    imgt = imnoise(imgt, 'salt & pepper');
    data = [data, imgt(:)];
    label = [label, this_label];
    
    imgt = imresize(imrotate(img, 15), [50,50]);
    data = [data, imgt(:)];
    label = [label, this_label];
    imgt = imnoise(imgt, 'salt & pepper');
    data = [data, imgt(:)];
    label = [label, this_label];
    
    imgt = imresize(imrotate(img, 20), [50,50]);
    data = [data, imgt(:)];
    label = [label, this_label];
    imgt = imnoise(imgt, 'salt & pepper');
    data = [data, imgt(:)];
    label = [label, this_label];
    
    imgt = imresize(imrotate(img, 25), [50,50]);
    data = [data, imgt(:)];
    label = [label, this_label];
    imgt = imnoise(imgt, 'salt & pepper');
    data = [data, imgt(:)];
    label = [label, this_label];
    
    imgt = imresize(imrotate(img, 30), [50,50]);
    data = [data, imgt(:)];
    label = [label, this_label];
    imgt = imnoise(imgt, 'salt & pepper');
    data = [data, imgt(:)];
    label = [label, this_label];
    
    imgt = imresize(imrotate(img, 350), [50,50]);
    data = [data, imgt(:)];
    label = [label, this_label];
    imgt = imnoise(imgt, 'salt & pepper');
    data = [data, imgt(:)];
    label = [label, this_label];
    
    imgt = imresize(imrotate(img, 340), [50,50]);
    data = [data, imgt(:)];
    label = [label, this_label];
    imgt = imnoise(imgt, 'salt & pepper');
    data = [data, imgt(:)];
    label = [label, this_label];
    
    imgt = imresize(imrotate(img, 330), [50,50]);
    data = [data, imgt(:)];
    label = [label, this_label];
    imgt = imnoise(imgt, 'salt & pepper');
    data = [data, imgt(:)];
    label = [label, this_label];
end

save('built_from_mis.mat', 'data', 'label');
