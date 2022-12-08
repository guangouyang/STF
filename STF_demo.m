clear;

%this data from the rare condition only

sub_no = 200;

epoch_twd = [-1000,2000];
baseline = [-200,0];

%ersp_data.mat can be downloaded from: https://www.dropbox.com/s/m74dolklyv88w7t/ersp_data.mat?dl=0
load ersp_data.mat

wt_ave = mean(ersp_all(:,:,:,:),4);

t_axis = linspace(epoch_twd(1),epoch_twd(2),size(wt_ave,2));
chs = {'Fz','Cz','Oz'};
b_lim = find(t_axis>baseline(1)&t_axis<baseline(2));
figure;
for j = 1:length(chs)
    ch = find(ismember({EEG.chanlocs.labels},chs{j}));
        temp = wt_ave(:,:,ch);
        subplot(1,length(chs),j);
        h = pcolor(t_axis,fs,temp-mean(temp(:,b_lim),2));set(h, 'EdgeColor', 'none');
        caxis([-0.7,0.7]);
        xlim([-200,1500]);
        colormap(jet);
        title(chs{j});
end


wt_data = wt_ave(:,:,:) - mean(wt_ave(:,b_lim,:),2);

comp_list = {[12,30,80,160],[12,30,320,420],[3,6,200,600],[7,14,200,700],[2,4,600,1000],[8,16,800,1400]};
ch_list = [17,17,24,18,17,18];
signs = [1,1,1,-1,1,1];
wt_data_i = zeros(size(wt_data,1),size(wt_data,2),size(wt_data,3),length(comp_list));

w_array_j = [];
for iter = 1:3
    disp(iter);
    for comp_i = 1:length(comp_list)
        wt_data_iter = wt_data;
        for sub_comp = 1:length(comp_list)
            if sub_comp ~= comp_i
                wt_data_iter = wt_data_iter - wt_data_i(:,:,:,sub_comp);
            end
        end


        f_p = find(fs<=comp_list{comp_i}(2)&fs>=comp_list{comp_i}(1));
        t_p = find(t_axis<=comp_list{comp_i}(4)&t_axis>=comp_list{comp_i}(3));

        wt_data_temp = wt_data_iter(f_p,t_p,ch_list(comp_i));
        [tmp1,tmp2] = max(wt_data_temp(:)*signs(comp_i));
        if iter>1 wt_data_temp = w_array_j(f_p,t_p,comp_i);
            [tmp1,tmp2] = max(wt_data_temp(:));end
        [f_tmp,t_tmp] = ind2sub(size(wt_data_temp),tmp2);
        f_p = f_p(f_tmp);
        t_p = t_p(t_tmp);

        epics(:,comp_i) = [f_p,t_p];


        temp = wt_data_iter(f_p,t_p,:);temp = temp(:);
        %         figure;topoplot(temp,EEG.chanlocs,'headrad','rim');axis([-.65,.65,-.65,.65]);
        vec = temp(:);vec_sign = signs(comp_i);

        pos = [];d_i = [];
        for f_i = 1:length(fs)
            for t_i = 1:length(t_axis)
                pos(:,end+1) = [f_i,t_i];
                d_i(size(pos,2)) = (f_i-f_p)^2 + (t_i-t_p)^2;
            end
        end
        [a,b] = sort(d_i);

        pos_sort = pos(:,b);

        w_array = zeros(length(fs),length(t_axis));
        w_array(f_p,t_p) = 1;ws = [];
        for j = 1:size(pos,2)
            xy = pos_sort(:,j);
            aa = squeeze(wt_data_iter(xy(1),xy(2),:));bb = vec;index = find(vec_sign*bb>0);
            ws(j) = mean(aa(index))/mean(bb(index));
            d = max(abs(xy(1) - f_p),abs(xy(2)-t_p));
            x_i = fix(linspace(f_p,xy(1),d+1));
            y_i = fix(linspace(t_p,xy(2),d+1));
            w_tmp = [];
            for k = 1:d+1
                w_tmp(k) = w_array(x_i(k),y_i(k));
            end
            if j>1
                if ws(j) >= min(w_tmp(1:end-1)) ws(j) = min(w_tmp(1:end-1));end
            end
            ws(ws<0) = 0;
            w_array(xy(1),xy(2)) = ws(j);
        end
        w_array_j(:,:,comp_i) = w_array;


        for jjj = 1:size(wt_data,3) wt_data_i(:,:,jjj,comp_i) = w_array.*vec(jjj);end
    end

%     temp = mean(wt_data_i(:,:,17,1),3);
%     figure;h = pcolor(t_axis,fs,temp);set(h, 'EdgeColor', 'none');colormap(jet);


end

figure;
for j = 1:length(comp_list)
    temp = wt_data_i(:,:,ch_list(j),j);
    subplot(2,3,j);
    h = pcolor(t_axis,fs,temp/max(abs(temp(:))));set(h, 'EdgeColor', 'none');colormap(jet);
    title([EEG.chanlocs(ch_list(j)).labels,'-',num2str(fs(epics(1,j))),'-',num2str(t_axis(epics(2,j)))]);
    xlim([-200,1500]);ylim([0.5,40]);
    caxis([-1,1]);
end




w_array_re = permute(w_array_j,[3,1,2]);
w_array_re = w_array_re(:,:);


topo_j = [];
tf_j = [];
for j = 1:sub_no
    disp(j);

    wt_data = ersp_all(:,:,:,j);
    wt_data = wt_data - mean(wt_data(:,b_lim,:),2);

    data = permute(wt_data,[3,1,2]);
    data = data(:,:);

    temp = (data*w_array_re')*inv(w_array_re*w_array_re');

    topo_j(:,:,j) = temp;



    temp1 =  (data'*temp)*inv(temp'*temp);

    temp2 = reshape(temp1,[71,451,6]);

    tf_j(:,:,:,j) = temp2;
end



figure;for j = 1:length(comp_list)
    subplot(2,3,j);
    h = pcolor(t_axis,fs,signs(j)*mean((tf_j(:,:,j,:)),4));set(h, 'EdgeColor', 'none');
    colormap(jet);caxis([-1,1]);xlim([-200,1500]);ylim([0.5,40]);
    title(['Cluster-',num2str(j)]);

end

temp = mean(topo_j,3);
figure;
for j = 1:length(comp_list)
    subplot(2,3,j);
topoplot(temp(:,j),EEG.chanlocs,'headrad','rim','style','map');axis([-.65,.65,-.65,.65]);
title(['Cluster-',num2str(j)]);
end


