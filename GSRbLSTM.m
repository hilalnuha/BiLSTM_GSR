close all
clc
clear

r=1;
L2=24;
load(['GSR' num2str(L2) '.mat'])

Xtr=GSRtr(:,1:L2)';
Ytr=GSRtr(:,L2+1)';
maxx=max(Ytr);

Xtr=GSRtr(:,1:L2)'/maxx;
Ytr=GSRtr(:,L2+1)'/maxx;


Xval=GSRval(:,1:L2)'/maxx;
Yval=GSRval(:,L2+1)'/maxx;

Xts=GSRts(:,1:L2)'/maxx;
Yts=GSRts(:,L2+1)'/maxx;

rng(r);

numiter=100;
numhid=20;
miniBatchSize = 800;
numFeatures=L2;
numResponses = 1;
numHiddenUnits = numhid;

layers = [ ... 
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

%layers = [ ...
%    sequenceInputLayer(numFeatures)
%    lstmLayer(numHiddenUnits,'OutputMode','sequence')
%    fullyConnectedLayer(numHiddenUnits)
%    dropoutLayer(0.1)
%    fullyConnectedLayer(numResponses)
%   regressionLayer];

maxEpochs = numiter;

options = trainingOptions('adam', ...
    MaxEpochs=maxEpochs, ...
    ValidationData={Xval,Yval}, ...
    ValidationFrequency=30, ...
    MiniBatchSize=miniBatchSize, ...
    InitialLearnRate=0.01, ...
    GradientThreshold=1, ...
    Verbose=0);

netModel= trainNetwork(Xtr,Ytr,layers,options);

%%
YtrOut = (predict(netModel,Xtr))*maxx;
Ytr=Ytr*maxx;
TrMse=mse(YtrOut,Ytr);
TrNmse=nmse(YtrOut,Ytr);
TrRmse=rmse(YtrOut,Ytr)
TrNrmse=nrmse(YtrOut,Ytr);
TrMae=mae(YtrOut,Ytr);
TrMbe=mbe(YtrOut,Ytr);
TrRsquare=rsquare(YtrOut',Ytr')

YvalOut = max(0,predict(netModel,Xval))*maxx;
Yval=Yval*maxx;
ValMse=mse(YvalOut,Yval);
ValNmse=nmse(YvalOut,Yval);
ValRmse=rmse(YvalOut,Yval)
ValNrmse=nrmse(YvalOut,Yval);
ValMae=mae(YvalOut,Yval);
ValMbe=mbe(YvalOut,Yval);
ValRsquare=rsquare(YvalOut',Yval')


YtsOut = max(0,predict(netModel,Xts))*maxx;
Yts=Yts*maxx;
TsMse=mse(YtsOut,Yts);
TsNmse=nmse(YtsOut,Yts);
TsRmse=rmse(YtsOut,Yts)
TsNrmse=nrmse(YtsOut,Yts);
TsMae=mae(YtsOut,Yts);
TsMbe=mbe(YtsOut,Yts);
TsRsquare=rsquare(YtsOut',Yts')



rang2=(1:200)+400;
plot(rang2,[Yts(rang2)' YtsOut(rang2)' ])
ylabel('GSR (W/m^2)')
xlabel('Hour')
grid
legend('Actual','Estimated','Location','northwest')

%%
figure
%rang2=(1:200)+400;

%plot(Yts(rang2),YtsOut(rang2), 'b*')
plot(Yts,YtsOut, 'b*')
  
ylabel('Actual')
xlabel('Predicted')
xlim([-50,1200])
ylim([-50,1200])
text(200,1000,0,strcat("R^2=",num2str(TsRsquare*100),"%"),'Color','k')
grid