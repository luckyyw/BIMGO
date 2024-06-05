

function [BestF,BestX,cnvg]=BIMGO(N,MaxIter,feat, label)

LB = 0;
UB = 1;
dim = size(feat, 2);
lb=ones(1,dim).*LB;  
ub=ones(1,dim).*UB;

%Initialize the first random population of Gazelles 
X=initialization(N,dim,UB,LB);
%Chaotic mapping
icmic=100;
for i=1:N
    for j=2:dim
        X(i,j)=sin(icmic/X(i,j-1));
    end
end
% initialize Best Gazelle
BestX=[];
BestFitness=inf;


for i=1:N   
    
    % Calculate the fitness of the population
    Sol_Cost(i,:)=fobj(X(i,:),feat, label);%#ok
    
    % Update the Best Gazelle if needed
    if Sol_Cost(i,:)<=BestFitness 
        BestFitness=Sol_Cost(i,:); 
        BestX=X(i,:);
    end
end

%mainloop
    for Iter=1:MaxIter
        for i=1:N
            
            RandomSolution=randperm(N,ceil(N/3));
            M=X(randi([(ceil(N/3)),N]),:)*floor(rand)+mean(X(RandomSolution,:)).*ceil(rand);     
            
            % Calculate the vector of coefficients
            cofi = Coefficient_Vector(dim,Iter,MaxIter);
            
            A=randn(1,dim).*exp(2-Iter*(2/MaxIter));
            D=(abs(X(i,:)) + abs(BestX))*(2*rand-1);
                       
            % Update the location
            NewX = Solution_Imp(X,BestX,lb,ub,N,cofi,M,A,D,i); 
            
            % Cost function calculation and Boundary check
            for j=1:4
                NewX(j,:) = boundaryCheck(NewX(j,:), LB, UB);
                Sol_CostNew(j,:)=fobj(NewX(j,:),feat, label);
            end
            % Adding new gazelles to the herd
            X=[X; NewX];       %#ok
            Sol_Cost=[Sol_Cost; Sol_CostNew];%#ok
            [~,idbest]=min(Sol_Cost);
            BestX=X(idbest,:);
            % Spiral disturbance
            L = 2*rand-1;
            z = exp(cos(pi*(1-(Iter/MaxIter))));
            X(i,:)= BestX+exp(z*L)*cos(2*pi*L)*abs(BestX-X(i,:));
        end
        
        % Update herd
        [Sol_Cost, SortOrder]=sort(Sol_Cost);
        X=X(SortOrder,:);
        [BestFitness,idbest]=min(Sol_Cost);   
        BestX=X(idbest,:);
        X=X(1:N,:);
        % Local search
        Alpha_pos = X(1,:);
        Beta_pos = X(2,:);
        LowB = -abs(Alpha_pos - Beta_pos);
        UpB = abs(Alpha_pos - Beta_pos);
        Temp = Beta_pos + rand(1,dim).*(UpB - LowB) + LowB;
        % Boundary check
        Flag4ub=Temp>ub;
        Flag4lb=Temp<lb;
        Temp=(Temp.*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb; 
        fitTemp = fobj(Temp(1, :),feat, label);
        if(fitTemp<BestFitness)
            BestFitness = fitTemp;
            BestX = Temp;
            X(1,:) = Temp;
        end
        Sol_Cost=Sol_Cost(1:N,:);
        cnvg(Iter)=BestFitness;%#ok
        BestX = sigmoidConversion(BestX);
        BestF=BestFitness;
    end
end


function errorRate = fobj(BestX, feat, label)
    Selection = sigmoidConversion(BestX);
    if sum(Selection == 1) == 0
        errorRate = 1;
    else
        sFeat = feat(:, Selection == 1);
        trainIdx = 1 : floor(0.8 * size(sFeat, 1));
        testIdx = floor(0.8 * size(sFeat, 1)) + 1 : size(sFeat, 1);
        xtrain = sFeat(trainIdx, :);
        ytrain = label(trainIdx);
        xvalid = sFeat(testIdx, :);
        yvalid = label(testIdx);
        % Training model
        My_Model = fitcsvm(xtrain, ytrain);
        % Prediction
        pred = predict(My_Model, xvalid);
        % Accuracy
        Acc = sum(pred == yvalid) / length(yvalid);
        % Error rate
        errorRate = 1 - Acc;
    end
end

% Binary conversion using Sigmoid function
function binary = sigmoidConversion(x)
    binary = sigmoid(x) > 0.5;
end

function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end
