close all;
clear all;
clc;
addpath('./Functions');

o = imread('./LAP1951e80/Holo.bmp');
o = rgb2gray(o);
o = im2double(o);
lap1=LAP(o);

a=[7.367724875065919e+02 2.722033279509556e+03 2.147610026910336e+03 3.510091503272092e+03 1.178414935791610e+04 1.224378820450961e+04]
lap = mapminmax(a, 0, 1);