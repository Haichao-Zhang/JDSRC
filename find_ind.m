function r_ind_set = find_ind(Corr,group,s)
%%=========================================================================  
%             Joint Dynamic Sparse Mapping
%                  by Haichao Zhang
%                 hczhang1@gmail.com
%                    Feb. 2012
%%=========================================================================

norm_type = 'l2'; 

task_num =  size(Corr,2);
group = group(:);

r_ind_set = [];

g_label = unique(group);
g_num = numel(g_label);
    sim_val = abs(Corr);

    r_ind = [];
    for t = 1:s

        r_val = zeros(g_num, size(Corr,2));

        for g = 1:g_num
            ind = find(group==g_label(g));
            [t_val,t_ind] = max(sim_val(ind,:),[],1);
            if(isempty(t_ind)||~sum(t_val~=0))
                %break;
                continue;
            end
            r_val(g,:) = t_val;
            r_ind(g,:) = t_ind;
            r_ind(g,:) = ind(r_ind(g,:));

        end
            
            if(strcmp(norm_type, 'l2'))
                [maxi,ind] = max(sum(r_val.^2,2));
            elseif (strcmp(norm_type, 'l1'))  
                [maxi,ind] = max(sum(abs(r_val),2));
            elseif (strcmp(norm_type, 'inf'))   
                [maxi,ind] = max(max(abs(r_val),[],2));
            end
    
    
    
        if(isempty( ind)||~sum(maxi~=0))
            break;

        end
 
    ind =(r_ind((ind),:));  
    r_ind_set = [r_ind_set;ind(:)'];
    

    for k = 1:task_num
      sim_val(ind(:,k),k)=0;
    end

    end