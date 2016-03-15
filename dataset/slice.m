% this script slices the images in ./raw_image into four parts and writes them to
% ./sliced, meanwhile, it prompts its user to label the image
% labeling can be an exhausting job :(
% remember, when labeling, enter '{label}' (with the quotation mark).

filenames = dir('raw_image');
for i = 199+3:1:length(filenames)

    img = im2double(imread(strcat('raw_image/',filenames(i).name)));
    img = double(~im2bw(img, 0.5));
    for j = 1:4
        img_slice = img(:, (j-1)*20+1:j*20);
        imshow(imresize(img_slice,[40,40]));
        try
            label = input(sprintf('input label for #%d: ', 4*(i-3)+j-1));
        catch ERR
            warning('ERR Occured, please try again.')
            label = input(sprintf('input label for #%d: ', 4*(i-3)+j-1));
        end
        close;
        imwrite(img_slice, strcat('sliced/', label, num2str(4*(i-3)+j-1), '.jpg'));
    end
end
