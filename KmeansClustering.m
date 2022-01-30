clear all
load mnist.mat
X = im2single(trainX);
k = 10; % K and p can be changed to satisfy problem
p = 0
while p < 1
    p = p+1;
    [c,z,jclust] = kmeanscluster(X,k);
    jcm(p) = min(jclust); %stores the mininum value of Jclust we get each time
    JC{p} = jclust;
    C{p} = c; 
    Z{p} = z;
end

%%
JC
C
Z
%%
jcm
[~,Imn]=min(jcm)
[~,Imx]=max(jcm)
%%
zmn = Z{Imn}%group representatives that correspond to the minimum jclust found
zmx = Z{Imx}%group representatives that correspond to the maximum jclust found
cmn = C{Imn}%labels that correspond to the minimum jclust found
cmx = C{Imx}%labels that correspond to the maximum jclust found
jcmn = JC{Imn}%Jclust values found each iteration for min jclust found
jcmx = JC{Imx}%Jclust values found each iteration for max jclust found
%%
%plot the value of the maximum jclust found and the minimum jclust found
figure
plot(1:length(jcmn),jcmn)
hold on 
plot(1:length(jcmx),jcmx)
legend("Jclust value minimum","Jclust value maximum")
xlabel("iteration")
ylabel("Jclust")
hold off
%%
%plotting the k group representantives for minimum Jclust
for i = 1:k
    grouprep = reshape(zmn(i,:),[28,28]);
    figure(2);
    subplot(5,k/5,i);
    imshow(grouprep');
    title([sprintf('%d',i),]);
end
figure(2)
%%
%plotting the k group representantives for maximum Jclust
for i = 1:k
    grouprep = reshape(zmx(i,:),[28,28]);
    figure(3);
    subplot(5,k/5,i);
    imshow(grouprep');
    title([sprintf('%d',i)]);
end
figure(3)
%%
%finding 10 nearest data points to z for minimum jclust
for t = 1:k
    for i = 1:length(X(:,1))
    nearmn(i) = norm(X(i,:)-zmn(t,:));
    end
    [~,in] = sort(nearmn);
    for l = 1:10
    b(l,:) = X(in(l),:);
    end
    near{t}=b;
end
%%
%plotting 10 nearest data points to z for min jclust
for t =1:k
    y = near{t};
    for i = 1:10
        im = reshape(y(i,:),[28,28]);
        figure(3+t);
        subplot(5,2,i);
        imshow(im');
        title(['data point ', sprintf('%d',i), ' for group ',sprintf('%d',t)]);
    end
end
%%
%finding 10 nearest data points to z for maximum jclust
for t = 1:k
    for i = 1:length(X(:,1))
    nearmx(i) = norm(X(i,:)-zmx(t,:));
    end
    [~,in2] = sort(nearmx);
    for l = 1:10
    d(l,:) = X(in2(l),:);
    end
    nearmax{t}=d;
end
%%
%plotting 10 nearest data points to z for max jclust
for t =1:k
    w = nearmax{t};
    for i = 1:10
        im2 = reshape(w(i,:),[28,28]);
        figure(3+k+t);
        subplot(5,2,i);
        imshow(im2');
        title(['data point ',sprintf('%d',i), ' for group ',sprintf('%d',t)]);
    end
end
%%
function [c,z,jclust] = kmeanscluster(X,k) %kmeans function (answer to part 1)
    c = randi(k,length(X(:,1)),1);%initialize and start off by assigning random labels to the data
    iter = 0;
    while iter < 15 %after testing out the function with the three different k values we were given, i concluded that they all converge after about 15 iterations
        iter = iter+1;
        for l = 1:k
            g{l} = find(c==l); % this gives the indices of xi where c == i and stores them in G{i}
            m = X(c==l,:);
            z(l,:) = (1/length(g{l}))*sum(m); %this is to find group representatives which is (1/(# of xi's in G{i}))*(all the xi's in G{i})   
        end
        %to find jclust value:
        for i = 1:length(X(:,1))
        b = X(i,:) - z(c(i),:);
        j(i) = norm(b)^2;
        end
        jclust(iter) = (1/length(X(:,1)))*sum(j);%jclust is (1/N)*(summation i=1-->N of norm(xi-zj)^2)
        %to find better fitting group:
        for i = 1:length(X(:,1))
            for t = 1:k
                fnc(t) = norm(X(i,:)-z(t,:))^2;
            end
            [a,ind] = min(fnc);
            c(i) = ind;
        end
    end %the loop iterates 15 times, and the c's and z's are updated every iteration. 
end