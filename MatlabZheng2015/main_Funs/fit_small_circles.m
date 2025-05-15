function [z, r, range] = fit_small_circles(sz, pixel_list, convex, boundary_points, R, factor, minPoints)
% FIT_SMALL_CIRCLES Fits small circles to convex boundary points
%
% Parameters:
%   sz - Size of the image [height, width]
%   pixel_list - List of pixel indices in the grain
%   convex - Array of convex boundary points [N x 2]
%   boundary_points - All boundary points [M x 2]
%   R - Maximum allowed circle radius
%   factor - Factor determining how close circle can be to boundary
%   minPoints - Minimum number of points required for circle fitting
%
% Returns:
%   z - Array of circle centers [2 x numCircles]
%   r - Array of circle radii [1 x numCircles]
%   range - Indices of points used for each circle [numCircles x 2]

% Initialize output arrays
z = [];
r = [];
range = [];

% Initialize working arrays
cv = convex;
fp = 1;                  % Index of first point in current subset
lp = length(cv);         % Index of last point in current subset

% Continue while we have enough points to fit a circle
while lp >= fp + minPoints
    % Try to fit a circle to the current subset of points
    [zc, rc] = fitcircle(cv(fp:lp, :)');
    
    % Calculate minimum distance from circle center to boundary
    min_dis = min(euclidian_distance(boundary_points, ones(size(boundary_points, 1),1)*zc'));
    
    % Calculate and check bounds for circle center
    zc_y = round(zc(2));
    zc_x = round(zc(1));
    coords_in_bounds = (zc_y >= 1 && zc_y <= sz(1) && zc_x >= 1 && zc_x <= sz(2));
    
    % If we have more than minimum points, we can try reducing the subset
    if lp > fp + minPoints
        % Check if circle is invalid for any reason
        if min_dis < rc || rc >= R ||...
           zc(2) < 1 || zc(2) > sz(1) ||...
           zc(1) < 1 || zc(1) > sz(2) ||...
           ~coords_in_bounds
            % If within bounds but not in pixel list
            if coords_in_bounds && ~any(pixel_list == sub2ind(sz, zc_y, zc_x))
                lp = lp - 1;
                continue;
            end
            % Any other invalid condition
            lp = lp - 1;
            continue;
        end
    % If were at the minimum number of points, move to the next starting point
    elseif lp == fp + minPoints
        % Check if circle is invalid for any reason
        if min_dis < factor*rc || rc >= R ||...
           zc(2) < 1 || zc(2) > sz(1) ||...
           zc(1) < 1 || zc(1) > sz(2) ||...
           ~coords_in_bounds
            % Try the next starting point
            fp = fp + 1;
            lp = length(cv);
            continue;
        end
        
        % Check if circle center is in the object
        if coords_in_bounds && ~any(pixel_list == sub2ind(sz, zc_y, zc_x))
            fp = fp + 1;
            lp = length(cv);
            continue;
        end
    end
    
    % If we reach here, the circle is valid - add it to results
    z = [z, zc];
    r = [r, rc];
    range = [range; fp, lp];
    
    % Move to the next segment
    fp = lp + 1;
    lp = length(cv);
end

end