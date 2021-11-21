function [C]=JDSR(X,y,s, task_ind, group, option)

%%=========================================================================  
%          Joint Dynamic Sparse Representation
%                  by Haichao Zhang
%                 hczhang1@gmail.com
%                     Feb. 2012
%%========================================================================= 
%  minimize_{C} \sum_k ||yk - Xk*C(:,k)||^2, s.t. ||C||_G \le s             
%%========================================================================= 
%                   Input Variables
%--------------------------------------------------------------------------
% X        :  the multi-observation(task) dictionary, meaning X = [X1; X2; ...; Xk; ...], with Xk \in R^{dk\times N}
% task_ind :  task_ind = [0, d1, d1+d2, ..., \sum_i di, ...] specifies the indecies of each task-specific-dictionary Xk in X
% y        :  test sample, signal = [y1; y2; ...; yk; ...], with y \in R^{dk}
% s        :  specified sparsity level
% group    :  length-N vector contains the lables for each column of X
% option   :  struct contains some additional parameters
%              option.lambdareg      -- regularization parameter
%              option.diffnorm       -- stop condition, norm differences of the residue
%              option.nbitermax      -- stop condition, maximum number of iterations
%--------------------------------------------------------------------------
%                   Output Variables
%--------------------------------------------------------------------------
% C        :  the JDSR matrix, C = [c1, c2, ..., ci, ...], with ci \in R^N
%%========================================================================= 
%%*************************************************************************
% Please cite the following papers if you find this work is useful:
%
% @inproceedings{JDSRC_ICCV11,
%   author    =       {Haichao Zhang and Nasser M. Nasrabadi and Yanning Zhang and Thomas S. Huang},
%   title     =       {Multi-observation Visual Recognition via Joint Dynamic Sparse Representation},
%   booktitle =       {IEEE International Conference on Computer Vision (ICCV)},
%   pages     =       {595-602},
%   year      =       {2011}
% }
%
% @article{JDSRC_PR12,
%   author    = {Haichao Zhang and Nasser M. Nasrabadi and Yanning Zhang and Thomas S. Huang},
%   title     = {Joint dynamic sparse representation for multi-view face recognition},
%   journal   = {Pattern Recognition},
%   volume    = {45},
%   number    = {4},
%   year      = {2012},
%   pages     = {1290-1298}
% }
%%*************************************************************************

lambdareg=option.lambdareg;


indiceA=[];
res=y;

nbiter=1;

task_num = numel(task_ind)-1;
K=task_num;

Cold=zeros(size(X,2),K);
C=Cold+1;
Corr = [];

    
    Avg_Res_Norm = 100;
    Avg_Res_Norm_old = Avg_Res_Norm+1;


while Avg_Res_Norm > option.diffnorm && nbiter < option.nbitermax  && (Avg_Res_Norm-Avg_Res_Norm_old)~=0
    

    Corr = [];
    for k =1:task_num
        inds = task_ind(k)+1:task_ind(k+1);
        Corr= [Corr X(inds,:)'*res(inds,:)];
    end
    

    r_ind_set = [];
    r_ind_set = find_ind(abs(Corr),group,2*s);

    indice_full = [r_ind_set;indiceA];
    
    beta=zeros(size(X,2),K);
    
    for k = 1:task_num
        indice = indice_full(:,k);
        inds = task_ind(k)+1:task_ind(k+1); 
        
        approx=(X(inds,indice)'*X(inds,indice)+ lambdareg*eye(length(indice)))\(X(inds,indice)'*y(inds)); 

        beta(indice,k)=approx;        
    end
    

    indiceA = find_ind(abs(beta),group,1*s); 
    
    
    Cold=C;
    C=zeros(size(X,2),K);



    for k = 1:task_num
        indice = indiceA(:,k);
        inds = task_ind(k)+1:task_ind(k+1); 
       
        approx=(X(inds,indice)'*X(inds,indice)+ lambdareg*eye(length(indice)))\(X(inds,indice)'*y(inds)); 

        C(indice,k)=approx;        
    end

    recon_sig = zeros(size(X,1),1);
    for k = 1:task_num 
        inds = task_ind(k)+1:task_ind(k+1);
        recon_sig(inds) = X(inds,:)*C(:,k);
    end
    res=y-recon_sig(:);

    
    nbiter=nbiter+1;
    

   Avg_Res_Norm_old = Avg_Res_Norm;
    Avg_Res_Norm = 0;
    for k = 1:task_num
        inds = task_ind(k)+1:task_ind(k+1);
        Avg_Res_Norm =  Avg_Res_Norm + norm(res(inds));
    end
    Avg_Res_Norm = Avg_Res_Norm/task_num;

end;

 C = zeros(size(X,2),K);

 Coef =inv_MTL(X,indiceA,y,task_ind);
 for k = 1:size(C,2)
     C(indiceA(:,k),k) = Coef(:,k);
 end


