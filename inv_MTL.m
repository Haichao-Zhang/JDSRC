function approx = inv_MTL(X,indice_full,signal,task_ind)

    for k = 1:numel(task_ind)-1
        inds = task_ind(k)+1:task_ind(k+1);
        indice = indice_full(:,k);
        approx(:,k)=pinv(X(inds,indice))*signal(inds);
    end