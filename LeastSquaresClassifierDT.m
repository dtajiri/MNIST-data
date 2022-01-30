clear all
load mnist.mat
A = im2single(trainX)%transforms A from unit8 to single in order to use for calculations
b = transpose(trainY)
N = length(b);
%in order to find minimum beta and alpha, I decided to let them be
%represented by one variable named theta. theta is a vecotr in the form [beta alpha]
a = ones([60000,1]); 
%Z matrix is my new form of A matrix. it is [A vectorof1s]
%Z*theta will result in the predictions
Z(:, [1:784,785]) = [A, a];
newZ = Z'*Z
%%
%one vs all classifier
k = [0:9];%there are 10 classes that have the label 0-9

%0 vs all
[theta0, pred0] = biclass(b,k(1),Z',newZ,Z);
print(theta0)
%%
%1 vs all
[theta1, pred1] = biclass(b,k(2),Z',newZ,Z);
%2 vs all
[theta2, pred2] = biclass(b,k(3),Z',newZ,Z);
%3 vs all
[theta3, pred3] = biclass(b,k(4),Z',newZ,Z);
%4 vs all
[theta4, pred4] = biclass(b,k(5),Z',newZ,Z);
%5 vs all
[theta5, pred5] = biclass(b,k(6),Z',newZ,Z);
%6 vs all
[theta6, pred6] = biclass(b,k(7),Z',newZ,Z);
%7 vs all
[theta7, pred7] = biclass(b,k(8),Z',newZ,Z);
%8 vs all
[theta8, pred8] = biclass(b,k(9),Z',newZ,Z);
%9 vs all
[theta9, pred9] = biclass(b,k(10),Z',newZ,Z);

%finding max of the 10 classifiers
Aova = [pred0,pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,pred9];
for i = 1:60000
    [fova(i), Ifova(i)] = max(Aova(i,:));
end
fova;
Ifova;
fhat1 = Ifova - 1; %fhat for one vs all classifer

Cmap = zeros(10);
for i = 1:60000 %this fills in the Cmap according to the data found
        if fhat1(i) == b(i)
            Cmap(b(i)+1,fhat1(i)+1) = Cmap(b(i)+1,fhat1(i)+1)+1;
        elseif fhat1(i) ~= b(i)
            Cmap(b(i)+1,fhat1(i)+1) = Cmap(b(i)+1,fhat1(i)+1)+1;
        end
end
p = sum(Cmap,2);
Cmap;
Cmap(:, [1:10,11]) = [Cmap, p];
d = sum(Cmap);
Cmap(11,:) = d
ovaerrate = (N-(Cmap(1,1)+Cmap(2,2)+Cmap(3,3)+Cmap(4,4)+Cmap(5,5)+Cmap(6,6)+Cmap(7,7)+Cmap(8,8)+Cmap(9,9)+Cmap(10,10)))/N %error rate is equal to (Ntotal-Ntrupostive)/Ntotal
%% 
% 

%one vs one classifier
%since K = 10, there will be a total of 45 classifiers

%0vs1 
[p01,th01] = ovoclass(b,0,1,Z,A); 
p01 = changevar(p01,0,1); % changes variable so that y = 1 is labeled 0 and y = -1 is labeled 0
%0vs2 
[p02,th02] = ovoclass(b,0,2,Z,A); 
p02 = changevar(p02,0,2); 
%0vs3 
[p03,th03] = ovoclass(b,0,3,Z,A); 
p03 = changevar(p03,0,3);
%0vs4 
[p04,th04] = ovoclass(b,0,4,Z,A); 
p04 = changevar(p04,0,4);
%0vs5 
[p05,th05] = ovoclass(b,0,5,Z,A); 
p05 = changevar(p05,0,5);
%0vs6 
[p06,th06] = ovoclass(b,0,6,Z,A); 
p06 = changevar(p06,0,6);
%0vs7 
[p07,th07] = ovoclass(b,0,7,Z,A); 
p07 = changevar(p07,0,7);
%0vs8 
[p08,th08] = ovoclass(b,0,8,Z,A); 
p08 = changevar(p08,0,8);
%0vs9 
[p09,th09] = ovoclass(b,0,9,Z,A); 
p09 = changevar(p09,0,9);
%1vs2 
[p12,th12] = ovoclass(b,1,2,Z,A); 
p12 = changevar(p12,1,2);
%1vs3 
[p13,th13] = ovoclass(b,1,3,Z,A); 
p13 = changevar(p13,1,3);
%1vs4 
[p14,th14] = ovoclass(b,1,4,Z,A); 
p14 = changevar(p14,1,4);
%1vs5 
[p15,th15] = ovoclass(b,1,5,Z,A); 
p15 = changevar(p15,1,5);
%1vs6 
[p16,th16] = ovoclass(b,1,6,Z,A); 
p16 = changevar(p16,1,6);
%1vs7 
[p17,th17] = ovoclass(b,1,7,Z,A); 
p17 = changevar(p17,1,7);
%1vs8 
[p18,th18] = ovoclass(b,1,8,Z,A); 
p18 = changevar(p18,1,8);
%1vs9 
[p19,th19] = ovoclass(b,1,9,Z,A); 
p19 = changevar(p19,1,9);
%2vs3 
[p23,th23] = ovoclass(b,2,3,Z,A); 
p23 = changevar(p23,2,3);
%2vs4 
[p24,th24] = ovoclass(b,2,4,Z,A); 
p24 = changevar(p24,2,4);
%2vs5 
[p25,th25] = ovoclass(b,2,5,Z,A); 
p25 = changevar(p25,2,5);
%2vs6 
[p26,th26] = ovoclass(b,2,6,Z,A); 
p26 = changevar(p26,2,6);
%2vs7 
[p27,th27] = ovoclass(b,2,7,Z,A); 
p27 = changevar(p27,2,7);
%2vs8 
[p28,th28] = ovoclass(b,2,8,Z,A); 
p28 = changevar(p28,2,8);
%2vs9 
[p29,th29] = ovoclass(b,2,9,Z,A); 
p29 = changevar(p29,2,9);
%3vs4 
[p34,th34] = ovoclass(b,3,4,Z,A); 
p34 = changevar(p34,3,4);
%3vs5 
[p35,th35] = ovoclass(b,3,5,Z,A); 
p35 = changevar(p35,3,5);
%3vs6 
[p36,th36] = ovoclass(b,3,6,Z,A); 
p36 = changevar(p36,3,6);
%3vs7 
[p37,th37] = ovoclass(b,3,7,Z,A); 
p37 = changevar(p37,3,7);
%3vs8 
[p38,th38] = ovoclass(b,3,8,Z,A); 
p38 = changevar(p38,3,8);
%3vs9 
[p39,th39] = ovoclass(b,3,9,Z,A); 
p39 = changevar(p39,3,9);
%4vs5 
[p45,th45] = ovoclass(b,4,5,Z,A); 
p45 = changevar(p45,4,5);
%4vs6 
[p46,th46] = ovoclass(b,4,6,Z,A); 
p46 = changevar(p46,4,6);
%4vs7 
[p47,th47] = ovoclass(b,4,7,Z,A); 
p47 = changevar(p47,4,7);
%4vs8 
[p48,th48] = ovoclass(b,4,8,Z,A); 
p48 = changevar(p48,4,8);
%4vs9 
[p49,th49] = ovoclass(b,4,9,Z,A); 
p49 = changevar(p49,4,9);
%5vs6 
[p56,th56] = ovoclass(b,5,6,Z,A); 
p56 = changevar(p56,5,6);
%5vs7 
[p57,th57] = ovoclass(b,5,7,Z,A); 
p57 = changevar(p57,5,7);
%5vs8
[p58,th58] = ovoclass(b,5,8,Z,A); 
p58 = changevar(p58,5,8);
%5vs9 
[p59,th59] = ovoclass(b,5,9,Z,A); 
p59 = changevar(p59,5,9);
%6vs7 
[p67,th67] = ovoclass(b,6,7,Z,A); 
p67 = changevar(p67,6,7);
%6vs8 
[p68,th68] = ovoclass(b,6,8,Z,A); 
p68 = changevar(p68,6,8);
%6vs9 
[p69,th69] = ovoclass(b,6,9,Z,A); 
p69 = changevar(p69,6,9);
%7vs8 
[p78,th78] = ovoclass(b,7,8,Z,A); 
p78 = changevar(p78,7,8);
%7vs9 
[p79,th79] = ovoclass(b,7,9,Z,A); 
p79 = changevar(p79,7,9);
%8vs9 
[p89,th89] = ovoclass(b,8,9,Z,A); 
p89 = changevar(p89,8,9);

Aovo = [p01',p02',p03',p04',p05',p06',p07',p08',p09',p12',p13',p14',p15',p16',p17',p18',p19',p23',p24',p25',p26',p27',p28',p29',p34',p35',p36',p37',p38',p39',p45',p46',p47',p48',p49',p56',p57',p58',p59',p67',p68',p69',p78',p79',p89'];
for i = 1:60000
    fhatovo(i) = mode(Aovo(i,:)); %gets the mode/value with most votes of the 45 classifiers
end

Cmapovo = zeros(10);
for i = 1:60000
        if fhatovo(i) == b(i)
            Cmapovo(b(i)+1,fhatovo(i)+1) = Cmapovo(b(i)+1,fhatovo(i)+1)+1;
        elseif fhatovo(i) ~= b(i)
            Cmapovo(b(i)+1,fhatovo(i)+1) = Cmapovo(b(i)+1,fhatovo(i)+1)+1;
        end
end
p = sum(Cmapovo,2);
Cmapovo(:, [1:10,11]) = [Cmapovo, p];
d = sum(Cmapovo);
Cmapovo(11,:) = d
ovoerrate = (N-(Cmapovo(1,1)+Cmapovo(2,2)+Cmapovo(3,3)+Cmapovo(4,4)+Cmapovo(5,5)+Cmapovo(6,6)+Cmapovo(7,7)+Cmapovo(8,8)+Cmapovo(9,9)+Cmapovo(10,10)))/N
%% 
% 
% 
% TEST TRAIN

Atrain = im2single(testX)
bt = transpose(testY)
at = ones([10000,1]); 
Ztest(:, [1:784,785]) = [Atrain, at]
Ztran = Ztest';
newZ2 = Ztran*Ztest;
n = length(bt);


%test one vs all classifier
k = [0:9];

%0 vs all
ft0 = Ztest*theta0; %multiply Z*theta to get predictions
%1 vs all
ft1 = Ztest*theta1;
%2 vs all
ft2 = Ztest*theta2;
%3 vs all
ft3 = Ztest*theta3;
%4 vs all
ft4 = Ztest*theta4;
%5 vs all
ft5 = Ztest*theta5;
%6 vs all
ft6 = Ztest*theta6;
%7 vs all
ft7 = Ztest*theta7;
%8 vs all
ft8 = Ztest*theta8;
%9 vs all
ft9 = Ztest*theta9;

%finding max of the 10 classifiers
Aovat = [ft0,ft1,ft2,ft3,ft4,ft5,ft6,ft7,ft8,ft9];
for i = 1:n
    [fovat(i), Ifovat(i)] = max(Aovat(i,:));
end
fhatt = Ifovat - 1; %overall prediction for one vs all classifer test

Cmapt = zeros(10);
for i = 1:n
        if fhat1(i) == bt(i)
            Cmapt(bt(i)+1,fhatt(i)+1) = Cmapt(bt(i)+1,fhatt(i)+1)+1;
        elseif fhat1(i) ~= bt(i)
            Cmapt(bt(i)+1,fhatt(i)+1) = Cmapt(bt(i)+1,fhatt(i)+1)+1;
        end
end
p = sum(Cmapt,2);
Cmapt(:, [1:10,11]) = [Cmapt, p];
d = sum(Cmapt);
Cmapt(11,:) = d
ovaerratet = (n-(Cmapt(1,1)+Cmapt(2,2)+Cmapt(3,3)+Cmapt(4,4)+Cmapt(5,5)+Cmapt(6,6)+Cmapt(7,7)+Cmapt(8,8)+Cmapt(9,9)+Cmapt(10,10)))/n

%one vs one classifier

%0vs1 
f01 = ovotest(Ztest,th01,0,1);
%0vs2 
f02 = ovotest(Ztest,th02,0,2);
%0vs3 
f03 = ovotest(Ztest,th03,0,3);
%0vs4 
f04 = ovotest(Ztest,th04,0,4);
%0vs5 
f05 = ovotest(Ztest,th05,0,5);
%0vs6 
f06 = ovotest(Ztest,th06,0,6);
%0vs7 
f07 = ovotest(Ztest,th03,0,7);
%0vs8 
f08 = ovotest(Ztest,th03,0,8);
%0vs9 
f09 = ovotest(Ztest,th09,0,9);
%1vs2 
f12 = ovotest(Ztest,th12,1,2);
%1vs3 
f13 = ovotest(Ztest,th13,1,3);
%1vs4 
f14 = ovotest(Ztest,th14,1,4);
%1vs5 
f15 = ovotest(Ztest,th15,1,5);
%1vs6 
f16 = ovotest(Ztest,th16,1,6);
%1vs7 
f17 = ovotest(Ztest,th17,1,7);
%1vs8 
f18 = ovotest(Ztest,th18,1,8);
%1vs9 
f19 = ovotest(Ztest,th19,1,9);
%2vs3 
f23 = ovotest(Ztest,th23,2,3);
%2vs4 
f24 = ovotest(Ztest,th24,2,4);
%2vs5 
f25 = ovotest(Ztest,th25,2,5);
%2vs6 
f26 = ovotest(Ztest,th26,2,6);
%2vs7 
f27 = ovotest(Ztest,th27,2,7);
%2vs8 
f28 = ovotest(Ztest,th28,2,8);
%2vs9 
f29 = ovotest(Ztest,th29,2,9);
%3vs4 
f34 = ovotest(Ztest,th34,3,4);
%3vs5 
f35 = ovotest(Ztest,th35,3,5);
%3vs6 
f36 = ovotest(Ztest,th36,3,6);
%3vs7 
f37 = ovotest(Ztest,th37,3,7);
%3vs8 
f38 = ovotest(Ztest,th38,3,8);
%3vs9 
f39 = ovotest(Ztest,th39,3,9);
%4vs5 
f45 = ovotest(Ztest,th45,4,5);
%4vs6 
f46 = ovotest(Ztest,th46,4,6);
%4vs7 
f47 = ovotest(Ztest,th47,4,7);
%4vs8 
f48 = ovotest(Ztest,th48,4,8);
%4vs9 
f49 = ovotest(Ztest,th49,4,9);
%5vs6 
f56 = ovotest(Ztest,th56,5,6);
%5vs7 
f57 = ovotest(Ztest,th57,5,7);
%5vs8
f58 = ovotest(Ztest,th58,5,8);
%5vs9 
f59 = ovotest(Ztest,th59,5,9);
%6vs7 
f67 = ovotest(Ztest,th67,6,7);
%6vs8 
f68 = ovotest(Ztest,th68,6,8);
%6vs9 
f69 = ovotest(Ztest,th69,6,9);
%7vs8 
f78 = ovotest(Ztest,th78,7,8);
%7vs9 
f79 = ovotest(Ztest,th79,7,9);
%8vs9 
f89 = ovotest(Ztest,th89,8,9);

Aovot = [f01',f02',f03',f04',f05',f06',f07',f08',f09',f12',f13',f14',f15',f16',f17',f18',f19',f23',f24',f25',f26',f27',f28',f29',f34',f35',f36',f37',f38',f39',f45',f46',f47',f48',f49',f56',f57',f58',f59',f67',f68',f69',f78',f79',f89'];
for i = 1:10000
    fhatovot(i) = mode(Aovot(i,:));
end

Cmapovot = zeros(10);
for i = 1:10000
        if fhatovot(i) == bt(i)
            Cmapovot(bt(i)+1,fhatovot(i)+1) = Cmapovot(bt(i)+1,fhatovot(i)+1)+1;
        elseif fhatovot(i) ~= bt(i)
            Cmapovot(bt(i)+1,fhatovot(i)+1) = Cmapovot(bt(i)+1,fhatovot(i)+1)+1;
        end
end
p = sum(Cmapovot,2);
Cmapovot(:, [1:10,11]) = [Cmapovot, p];
d = sum(Cmapovot);
Cmapovot(11,:) = d
ovoerratet = (n-(Cmapovot(1,1)+Cmapovot(2,2)+Cmapovot(3,3)+Cmapovot(4,4)+Cmapovot(5,5)+Cmapovot(6,6)+Cmapovot(7,7)+Cmapovot(8,8)+Cmapovot(9,9)+Cmapovot(10,10)))/n
%%
function [theta,prediction] = biclass(b,k,Zt,newZ,Z)%finds theta for one vs all classifier
y(b == k) = 1
y(b ~= k) = -1
newy = Zt*y'
theta = pinv(newZ)*newy %theta(found by using least mean squares) is a 785x1 vector that is composed of beta and alpha. beta is the first 784 elements of theta and alpha is the 785th element
prediction = Z*theta
end

function l = changevar(y,yes,no)
    l(y == 1) = yes;
    l(y == -1) = no;
end

function f = s(prediction) %gets the sign(a) where a>=0  is 1 and a<0 is -1
    f = sign(prediction);
    f(f == 0) = 1;
end

function [p,th] = ovoclass(b,one,one2,Z,A) %finds theta(using least mean squares) and prediction for one vs one classifier
%the following is the process of getting a subset data thats composed of
%only i and j values
y = b;
y(y ~= one & y ~= one2) = 10; 
j = find(y ~= 10);
At = A(j,:);
y = y(y~=10);
ny(y == one2) = -1;
ny(y == one) = 1;
%the following is my approach to getting the least mean square
a = ones([length(j),1]); 
At(:, [1:784,785]) = [At, a];
Asub = At'*At;
nb = At'*ny';
th = pinv(Asub)*nb;
pred = Z*th;%Matrix Z*theta gives my f tilda
p = s(pred);%this gives me fhat = sign(ftilda)
end

function f = ovotest(Ztest,th,one,one2)%used to test the one vs one classifier. Uses the trained theta to predict an outcome for the test data
    f = Ztest*th;
    f = s(f);
    f = changevar(f,one,one2);
end