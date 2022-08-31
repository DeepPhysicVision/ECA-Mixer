% wavelength = 636,530,470
close all;
clear all;
clc;
addpath('./Functions');

% change 1
o = imread('001Z1/0.bmp');
% o = rgb2gray(o);
o = mat2gray(o);
q = o;

%% ASA Paprameters
% change 2
lambda = 636e-6;  % wavelength (um)
z = 1;
detector_size = 1.85e-3;  % pixel pitch (um)

nx = size(q,1);  % data size
ny = size(q,2);
nz = 1;
k = 2*pi/lambda;
sensor_size = nx*detector_size;  % detector size (um)
deltaX = detector_size;
deltaY = detector_size;
Nx = nx;
Ny = ny*nz*2;
Nz = 1;
Phase=MyMakingPhase(nx,ny,z,lambda,deltaX,deltaY);
E0=ones(nx,ny);  
E=MyFieldsPropagation(E0,nx,ny,nz,Phase);  
g=q;
g=MyC2V(g(:));
transf=MyAdjointOperatorPropagation(g,E,nx,ny,nz,Phase);
transf=reshape(MyV2C(transf),nx,ny,nz);

A = @(f_twist) MyForwardOperatorPropagation(f_twist,E,nx,ny,nz,Phase);  % forward propagation operator
AT = @(g) MyAdjointOperatorPropagation(g,E,nx,ny,nz,Phase);  % backward propagation operator

%% TwIST algorithm (8)
tic

tau=0.04;
piter = 4;
tolA = 1e-6;
iterations = 500;

Psi = @(f,th) MyTVpsi(f,th,0.05,piter,Nx,Ny,Nz);
Phi = @(f) MyTVphi(f,Nx,Ny,Nz);

[f_reconstruct,dummy,obj_twist,...
    times_twist,dummy,mse_twist]= ...
    TwIST(g,A,tau,...
    'AT', AT, ...
    'Psi', Psi, ...
    'Phi',Phi, ...
    'Initialization',2,...
    'Monotone',1,...
    'StopCriterion',1,...
    'MaxIterA',iterations,...
    'MinIterA',iterations,...
    'ToleranceA',tolA,...
    'Verbose', 1);
f_reconstruct=reshape(MyV2C(f_reconstruct),nx,ny,nz);

% change 3
amp=(abs(f_reconstruct));
amp = im2double(amp);
imwrite(amp,'./001Z1/Amplitude/Best_0.bmp');

pha = (angle(f_reconstruct));
pha = unwrap(pha);
pha = im2double(pha);
imwrite(pha,'./001Z1/Phase/Best_0.bmp');

toc

