fprintf(1,['\n\n', ...
' # c)  Multiple running the Human Pose demo:\n']);
%pause
clear; close all;

for i = 3:3
%     initVardistItersList = [100 200 300 400 500];
%     itNoList = [1000 2000 3000 4000 5000];
    experimentNo = i;
    initial_X = 'concatenated';
%     initVardistIters = initVardistItersList(i-15);
%     itNo = itNoList(i-15);
    dynamicKern = {'rbf', 'white', 'bias'}; % SEE KERN toolbox for kernels\n', ...
    dynamicsConstrainType = {'time'};
    demHumanPoseSvargplvm1;
    clear;
end
