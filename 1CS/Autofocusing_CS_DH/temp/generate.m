close all;
clear all;clc;
addpath('./Functions');

o = imread('peppers.tif');
%o = rgb2gray(o);
o = im2double(o);
f=o;
%figure;imshow(abs(f));title('Object')

%% Paprameters (1)
nx=size(f,1);  % data size
ny=size(f,2);
nz=1;
lambda=0.635;  % wavelength (um)
k = 2*pi/lambda;
detector_size=2;  % pixel pitch (um)
sensor_size=nx*detector_size;  % detector size (um)
z=1000;  % distance from detector to first reconstructed plane (um)
deltaX=detector_size;
deltaY=detector_size;
Nx=nx;
Ny=ny*nz*2;
Nz=1;

%% Propagation kernel (2)
Phase=MyMakingPhase(nx,ny,z,lambda,deltaX,deltaY);
%figure;imagesc(plotdatacube(angle(Phase)));title('Phase of kernel');axis image;drawnow;
axis off; colormap(hot); colorbar;
E0=ones(nx,ny);  % illumination light
E=MyFieldsPropagation(E0,nx,ny,nz,Phase);  % propagation of illumination light
%% Field measurement and backpropagation (3)
cEs=zeros(nx,ny,nz);
Es=f.*E;
for i=1:nz
    cEs(:,:,i)=fftshift(fft2(Es(:,:,i)));
end
cEsp=sum(cEs.*Phase,3);
S=(ifft2(ifftshift(cEsp)));

f1 = ones(nx,ny);
Es1=f1.*E;
for i=1:nz
    cEs1(:,:,i)=fftshift(fft2(Es1(:,:,i)));
end
cEsp1=sum(cEs1.*Phase,3);
S1=(ifft2(ifftshift(cEsp1)));
s=(S+1).*conj(S+1);
s1=(S1+1).*conj(S1+1);
%  diffracted field
q = s./s1; % normalized 
q = im2double(q);
figure;imshow(abs(q));title('Diffracted field')

aa = abs(q);
bb = mat2gray(abs(q));

imwrite(abs(q),'Ht.png')
% imwrite(mat2gray(abs(q)), 'temp.png')



