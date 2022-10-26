% Neural Network 
clc;
clear all;

df = readtable('alldata_p.csv');
df = df{:,:};
ndvi = df(:,2);
gndvi =  df(:,3);
osavi = df(:,4);
lci = df(:,5);
ndre = df(:,6);
swp = df(:,7);
lt = df(:,8);

swp_lt = [lt swp];
input0 = [ndvi gndvi osavi lci ndre lt];
input1 = [ndvi gndvi osavi lci ndre];
input2 = [ndvi gndvi ndre lt];
input3 = [osavi];

% net = feedforwardnet;
% net.numinputs = 2;
% net = configure(net,x);
% net = train(net,x,t);
% view(net)
