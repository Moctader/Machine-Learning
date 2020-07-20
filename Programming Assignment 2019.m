%% TITLE ****************************************************************
% *                                                                      *
% *              		 521289S Machine Learning 					     *
% *                     Programming Assignment 2019                      *
% *                                                                      *
% *   Author 1: << Md Golam Moctader and 2601656 >>             *
% *                                                                      *
% *   NOTE: The file name for this file MUST BE 'classify.m'!            *
% *         Everything should be included in this single file.           *
% *                                                                      *
% ************************************************************************     


%%
function classify()
load('trainingData.mat')
data_training  = trainingData;
all_Classes = class_trainingData;
%whitening
data_training  = standardize(data_training );
N = length(all_Classes);

    %% Splitting the data
    rng(7) % to see how changes we made changed the validation_result we fix the 'seed' to get the same random ordering of number
    selection = randperm(N); % see also 'help rng'
    training_data = data_training (selection(1:floor(2*N/3)), :);
    validation_data = data_training(selection((floor(2*N/3)+1):N), :);
    
    training_class = all_Classes(selection(1:floor(2*N/3)), 1);
    validation_class = all_Classes(selection((floor(2*N/3)+1):N), 1);
    
    
    % resV : The accuracy vector
    % bestFSet : The most relevant features 
    [resV, bestFSet] = SFFS(data_training,all_Classes)


end


%% PUBLIC INTERFACE ******************************************************
%%
function nick = getNickName()
    nick = 'ranking';   % CHANGE THIS!
end


%% This is the training interface for the classifier you are constructing.

%%
function parameters = trainClassifier( samples, classes )
%%

    N = length(classes);
%% Splitting the data
    selection = randperm(N); % see also 'help rng'
    training_data = samples(selection(1:floor(2*N/3)), :);
    validation_data = samples(selection((floor(2*N/3)+1):N), :);
    training_class = classes(selection(1:floor(2*N/3)), 1);
    validation_class = classes(selection((floor(2*N/3)+1):N), 1);
    
    
    
    [resV, bestFSet] = SFFS(training_data,training_class)
 %parameters = struc(bestFSet, training_data, training_class);
    parameters = [ [ bestFSet; training_data ] , [0; training_class] ];


end


%% This is the evaluation interface of your classifier.

%%
function results = evaluateClassifier( samples, parameters )

[m,n] = size(parameters);
    training_data = parameters(2:m, 1:n-1);
    best_featurevector = parameters(1, 1:n-1);
    training_class = parameters(2:m, n);
    
results = knnclass(samples, training_data, best_featurevector, training_class, 3);
end

%% 
function [feat_out] = standardize(feat_in)
N = size(feat_in,1); 
% centering
center = feat_in-repmat(mean(feat_in), N, 1);
% standardization
feat_stand = center./repmat(std(center), N, 1);
 
% whitening eigenvalue decomposition
[V,D] = eig(cov(center)); %see help eig
W = sqrt(inv(D)) * V' ;
z=W* center'; 
feat_whit2 = z';
feat_out = feat_whit2; 
end
 
% sffs
function [res_vector, best_fset] = SFFS(data,data_c)
% Vector of used features
fv = zeros(1,size(data, 2)); 
max_n_feat = length(fv); 
% Initial dimension is one
n_features = 1;
best_result = 0;
res_vector = zeros(1,max_n_feat);  
search_direction = 0;  
k = 1; 
 
while(n_features <= max_n_feat)
    [best_result_add, best_feature_add] = findbest(data, data_c, fv, search_direction,k); 
    fv(best_feature_add) = 1;
    
    if(best_result < best_result_add)
       best_result = best_result_add;
       best_fset = fv;
    end
 
    if(best_result_add > res_vector(n_features)) 
        res_vector(n_features) = best_result_add;
    end
        
    %print current result
    disp([res_vector(n_features) n_features])
 
    search_direction = 1;
 
    while search_direction
          
            if(n_features > 2)
                % Search the worst feature
                [best_result_rem, best_feature_rem] = findbest(data, data_c, fv, search_direction,k);
                % If better than before, step backwards and update results
                % otherwise we will go to the inclusion step
                if(best_result_rem > res_vector(n_features - 1))
                    fv(best_feature_rem) = 0;
                    n_features = n_features - 1;
                    if(best_result < best_result_rem)
                        best_result = best_result_rem;
                        best_fset = fv;
                    end
                
                    res_vector(n_features) = best_result_rem;
                
                    %print current result
                    disp([res_vector(n_features) n_features])
                
                else
                    search_direction = 0;
                end
            
            else
                search_direction = 0;
            end
        
     end
      n_features = n_features + 1;
    
end
 

 
res_vector
best_fset
 
end
 
% feature selection
function [best, feature] = findbest(data, data_c, fvector, direction, k)
 
num_samples = length(data);
best = 0;
feature = 0;
 
for in = 1:length(fvector)
    if (direction == 0 && fvector(in) == 0 ||direction == 1 && fvector(in) == 1 )
        if direction == 0 
            fvector(in) = 1; % here we set the vector element corresponding to 'in' as one; 'fvector' is then used as a mask to choose the coulmns from the data matrix which are ones in 'fvector'
        else  
            fvector(in) = 0;
        end
      
        D = squareform( pdist( data(:,logical(fvector)) ) ); 
       
        [D, I] = sort(D, 1);
        I = I(1:k+1, :);
        labels = data_c( : )';
        if k == 1 
            predictedLabels = labels( I(2, : ) )';
        else 
            predictedLabels = mode( labels( I( 1+(1:k), : ) ), 1)';
        end
        correct = sum(predictedLabels == data_c); 
        result = correct/num_samples; 
        if(result > best) 
            best = result;
            feature = in; 
        end
       
        if direction == 0
            fvector(in) = 0; 
        else
            fvector(in) = 1;
        end
    end
end
 
end
 
% Knn
function [predictedLabels] = knnclass(dat1, dat2, fvec, classes, k)
 
p1 = pdist2( dat1(:,logical(fvec)), dat2(:,logical(fvec)) );
% Here we aim in finding k-smallest elements
[D, I] = sort(p1', 1);
I = I(1:k+1, :);
labels = classes( : )';
if k == 1 
    predictedLabels = labels( I(2, : ) )';
else 
    predictedLabels = mode( labels( I( 1+(1:k), : ) ), 1)'; 
end
 
end
 













