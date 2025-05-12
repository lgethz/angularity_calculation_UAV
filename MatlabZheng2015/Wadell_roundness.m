function Wadell_roundness(input_dir, csv_file, tol, factor, span, exclusion_range)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODIFICATIONS:
% This is a modified version of Junxing Zhengs original code.
% Modifications by Lars Gübeli ETH Zürich (2025):
% - Added functionality to handle overlap points between grains
% - Modified to accept input_dir and csv_file parameters for integration with Python
% - Added filtering of convex points near overlap regions to improve angularity calculation
% - Enhanced visualization to show overlap regions
%
% ORIGINAL CODE:
% Copyright (c) 2016, Junxing Zheng
% All rights reserved.
%
% Original author:
%       Junxing Zheng  (Sep. 2014)
%       University of Michigan, Ann Arbor
%       junxing@umich.edu 
%
% Citation:
%   Zheng, J., and Hryciw, R.D. (2015). Traditional Soil Particle Sphericity, 
%   Roundness and Surface Roughness by Computational Geometry, 
%   Geotechnique, Vol. 65, No. 6, 494-506, DOI:10.1680/geot./14-P-192.
%
% LICENSE:
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
%    * Redistributions of source code must retain the above copyright
%      notice, this list of conditions and the following disclaimer.
%    * Redistributions in binary form must reproduce the above copyright
%      notice, this list of conditions and the following disclaimer in
%      the documentation and/or other materials provided with the distribution
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all
addpath(genpath('main_Funs'))

% Add parameter validation with defaults
if nargin < 3 || isempty(tol)
    tol = 0.05;  % Default value
end
if nargin < 4 || isempty(factor)
    factor = 0.98;  % Default value
end
if nargin < 5 || isempty(span)
    span = 0.07;  % Default value
end
if nargin < 6 || isempty(exclusion_range)
    exclusion_range = 22;  % Default value
end


%% Create output directory for results
output_dir = fullfile(fileparts(input_dir), 'analysis_results');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Load overlap points from CSV
try
    overlap_data = readtable(csv_file);
    fprintf('Loaded %d overlap points from %s\n', height(overlap_data), csv_file);
catch e
    fprintf('Warning: Could not load overlap points: %s\n', e.message);
    overlap_data = table([], [], [], 'VariableNames', {'GrainID', 'X', 'Y'});
end

%% Get all grain image files
grain_files = dir(fullfile(input_dir, 'grain_*.jpg'));
fprintf('Found %d grain images to process\n', length(grain_files));

%% Initialize results array
all_results = [];

%% Process each grain image
for img_idx = 1:length(grain_files)
    % Get grain ID from filename
    filename = grain_files(img_idx).name;
    [~, name, ~] = fileparts(filename);
    parts = split(name, '_');
    grain_id = str2double(parts{2});
    
    fprintf('Processing grain %d (%s)...\n', grain_id, filename);
    
    % Get overlap points for this grain
    grain_overlap = overlap_data(overlap_data.GrainID == grain_id, :);
    
    % Read and process image
    img_path = fullfile(input_dir, filename);
    img = imread(img_path);
    
    % Convert to binary if needed
    if size(img, 3) > 1
        img = rgb2gray(img);
    end
    level = graythresh(img);
    im = im2bw(img, level);
    BW = ~im;  % Were assuming white grain on black background
    
    % Standard processing
    dist_map = bwdist(~BW);
    sz = size(BW);
    cc = bwconncomp(BW, 8);
    
    if cc.NumObjects == 0
        fprintf('  No objects detected, skipping\n');
        continue;
    end

    particles = discrete_boundary(cc);
    particles = nonparametric_fitting(particles, span);
    
    % Create figure for visualization
    f = figure;
    imshow(img);
    hold on;
    
    % Process the grain
    obj = cc.PixelIdxList{1};  % Just process the first object
    [R, RInd] = max(dist_map(obj));
    [cy, cx] = ind2sub(sz, obj(RInd));
    boundary_points = particles.objects(1).cartesian;
    X = boundary_points(:, 1);
    Y = boundary_points(:, 2);
    
    % Plot largest circle
    theta = linspace(0, 2*pi, 100);
    h_inscribed = plot(cos(theta)*R+cx, sin(theta)*R+cy, 'm', 'LineWidth', 2);

    % Segment the boundary
    seglist = segment_boundary(X, Y, tol, 0);

    % Concave and convex detection
    [concave, convex] = concave_convex(seglist, [cx, cy], 0);


filtered_convex = convex;  % Start with all convex points
h_overlap = []; % Initialize to empty

% Add debug output to show number of overlap points
fprintf('  Found %d overlap points for grain %d\n', height(grain_overlap), grain_id);

if ~isempty(grain_overlap) && height(grain_overlap) > 0
    % Plot overlap points - make them smaller (30 instead of 50)
    h_overlap = scatter(grain_overlap.X, grain_overlap.Y, 30, 'ro', 'filled');
    
    % Filter out convex points near overlap regions
    filtered_convex = [];
    excluded_convex = []; % Store the excluded points
    
    for i = 1:size(convex, 1)
        px = convex(i, 1);
        py = convex(i, 2);
        
        % Check if this convex point is near any overlap point
        is_overlap = false;
        for j = 1:height(grain_overlap)
            ox = grain_overlap.X(j);
            oy = grain_overlap.Y(j);
            
            % If point is within 5 pixels of an overlap point, exclude it
            if sqrt((px-ox)^2 + (py-oy)^2) < exclusion_range
                is_overlap = true;
                break;
            end
        end

        % Keep only non-overlap points, track excluded ones
        if ~is_overlap
            filtered_convex = [filtered_convex; convex(i,:)];
        else
            excluded_convex = [excluded_convex; convex(i,:)];
        end
    end
    
    fprintf('  Filtered out %d of %d convex points near overlaps\n', ...
            size(excluded_convex,1), size(convex,1));
else
    fprintf('  No overlap points found for this grain, using all convex points\n');
    excluded_convex = []; % No excluded points in this case
end

% Plot corner points - used points in blue, excluded in orange
if isempty(filtered_convex)
    fprintf('  Warning: All convex points were filtered out. Skipping this grain.\n');
    continue;  % Skip to the next grain
end

% After the empty check, add a minimum count check
if size(filtered_convex, 1) < 8  % At least 8 points needed for reliable processing
    fprintf('  Warning: Too few convex points (%d) remain after filtering. Skipping this grain.\n', size(filtered_convex, 1));
    continue;  % Skip to the next grain
end

h_corners = plot(filtered_convex(:,1), filtered_convex(:,2), 'b.', 'MarkerSize', 10);
h_excluded = [];
if ~isempty(excluded_convex)
    h_excluded = plot(excluded_convex(:,1), excluded_convex(:,2), 'o', 'MarkerSize', 8, 'Color', [1 0.5 0]); % Orange color
end

    % Fit small circles (only to non-overlap points)
    try
        [z, r] = compute_corner_circles(sz, obj, filtered_convex, boundary_points, R, factor, 3);
        
        if isempty(z) || isempty(r)
            fprintf('  Warning: No valid corner circles found. Skipping this grain.\n');
            continue;  % Skip to the next grain
        end
    catch e
        fprintf('  Error in corner circle computation: %s. Skipping this grain.\n', e.message);
        continue;  % Skip to the next grain
    end

    % Plot the circle centers and circles
    h_centers_all = [];
    h_small_circles_all = [];
    for ee = 1:length(r)
        h_center = plot(z(ee, 1), z(ee, 2), 'g*', 'MarkerSize', 5);
        h_circle = plot(z(ee, 1) + r(ee) * cos(theta), ...
             z(ee, 2) + r(ee) * sin(theta), 'g', 'LineWidth', 2);
        
        % Store first handles for legend
        if ee == 1
            h_centers_all = h_center;
            h_small_circles_all = h_circle;
        end
    end

    % Create legend items array with valid handles only
legend_items = {};
legend_labels = {};

% Only add items that exist
if ~isempty(h_inscribed)
    legend_items{end+1} = h_inscribed;
    legend_labels{end+1} = 'Largest inscribed circle';
end

if ~isempty(h_corners)
    legend_items{end+1} = h_corners;
    legend_labels{end+1} = 'Used corner points';
end

if ~isempty(h_excluded)
    legend_items{end+1} = h_excluded;
    legend_labels{end+1} = 'Excluded corner points';
end

if ~isempty(h_centers_all)
    legend_items{end+1} = h_centers_all;
    legend_labels{end+1} = 'Circle centers';
end

if ~isempty(h_small_circles_all)
    legend_items{end+1} = h_small_circles_all;
    legend_labels{end+1} = 'Small circles';
end

% Add overlap points to legend if they exist
if ~isempty(h_overlap)
    legend_items{end+1} = h_overlap;
    legend_labels{end+1} = 'Overlap points';
end

    % Create the legend - use cell arrays properly
    lgd = legend([legend_items{:}], legend_labels, 'Location', 'southoutside', 'Orientation', 'horizontal');
    set(lgd, 'FontSize', 8);

    % Calculate roundness
    Roundness = mean(r)/R;
    if Roundness > 1
        Roundness = 1;
    end
    
    % Add grain ID and results to the plot
    title_str = sprintf('Grain %d - Roundness: %.3f', grain_id, Roundness);
    title(title_str, 'FontSize', 12);   
    
    % Sphericity computation
    [~, rcum] = min_circum_circle(X, Y);
    sphericity1 = particles.objects(1).area/(pi*rcum^2);  % area sphericity
    sphericity2 = sqrt(particles.objects(1).area/pi)/rcum;   % diameter sphericity
    sphericity3 = R/rcum;   % circle ratio sphericity
    sphericity4 = 2*sqrt(pi*particles.objects(1).area)/particles.objects(1).perimeter; % perimeter sphericity
    sphericity5 = particles.objects(1).d1d2(2)/particles.objects(1).d1d2(1); % width to length ratio sphericity
    
    % Add result to array
    all_results = [all_results; grain_id Roundness sphericity1 sphericity2 sphericity3 sphericity4 sphericity5];
    
    % Save the figure
    output_file = fullfile(output_dir, sprintf('grain_%d_analysis.png', grain_id));
    saveas(f, output_file);
    fprintf('  Saved analysis image to %s\n', output_file);
    close(f);
end

%% Save all results to CSV
if ~isempty(all_results)
    % Create results table
    result_table = array2table(all_results, 'VariableNames', ...
        {'GrainID', 'Roundness', 'AreaSphericity', 'DiameterSphericity', ...
         'CircleRatioSphericity', 'PerimeterSphericity', 'WidthLengthRatioSphericity'});
    
    % Save to CSV
    csv_output = fullfile(output_dir, 'roundness_results.csv');
    writetable(result_table, csv_output);
    fprintf('Saved all results to %s\n', csv_output);
    
    % Display summary with NaN handling
    fprintf('\nSummary of Results:\n');
    
    % Get valid (non-NaN) values for each metric
    valid_roundness = all_results(~isnan(all_results(:,2)), 2);
    valid_area_sph = all_results(~isnan(all_results(:,3)), 3);
    valid_diam_sph = all_results(~isnan(all_results(:,4)), 4);
    
    % Report averages with count information
    if ~isempty(valid_roundness)
        fprintf('Average Roundness: %.3f (calculated from %d of %d grains)\n', ...
            mean(valid_roundness), length(valid_roundness), size(all_results,1));
    else
        fprintf('Average Roundness: N/A (no valid measurements)\n');
    end
    
    if ~isempty(valid_area_sph)
        fprintf('Average Area Sphericity: %.3f (calculated from %d of %d grains)\n', ...
            mean(valid_area_sph), length(valid_area_sph), size(all_results,1));
    else
        fprintf('Average Area Sphericity: N/A (no valid measurements)\n');
    end
    
    if ~isempty(valid_diam_sph)
        fprintf('Average Diameter Sphericity: %.3f (calculated from %d of %d grains)\n', ...
            mean(valid_diam_sph), length(valid_diam_sph), size(all_results,1));
    else
        fprintf('Average Diameter Sphericity: N/A (no valid measurements)\n');
    end
end