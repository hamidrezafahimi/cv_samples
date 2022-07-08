I = imread('fig2.jpg');
J = rgb2gray(I);

responses(:,:,1) = conv2(J,ans(:,:,1),'valid');
q = size(ans);
p = size(responses);
res = zeros(p(1), p(2));
for i=1:q(3)
    responses(:,:,i)=conv2(J,ans(:,:,i),'valid');
    res = res + responses(:,:,i);
end


% imshow(responses(:,:,6));
imshow(res);