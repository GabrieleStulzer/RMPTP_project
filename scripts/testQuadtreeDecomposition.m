function testQuadtreeDecomposition()
% TESTQUADTREEDECOMPOSITION Test script for quadtree decomposition with four rectangular obstacles.

    % Clear workspace and figures to avoid conflicts
    clear all
    close all
    clc

    % Parameters
    gridSize = 256; % Define the size of the grid (e.g., 256x256)
    
    % Initialize grid with zeros
    grid = zeros(gridSize, gridSize);
    
    % Define four rectangular obstacles
    % Each obstacle is defined by [x_start, y_start, width, height]
    obstacles = [
        50, 50, 40, 60;    % Obstacle 1
        150, 30, 30, 80;   % Obstacle 2
        80, 150, 60, 40;   % Obstacle 3
        180, 180, 50, 50    % Obstacle 4
    ];
    
    % Add obstacles to the grid
    for i = 1:size(obstacles, 1)
        x_start = obstacles(i, 1);
        y_start = obstacles(i, 2);
        width = obstacles(i, 3);
        height = obstacles(i, 4);
        
        x_end = min(x_start + width - 1, gridSize);
        y_end = min(y_start + height - 1, gridSize);
        
        grid(y_start:y_end, x_start:x_end) = 1;
    end
    
    % Convert grid to double (if not already)
    grid = double(grid);
    
    % Display grid type and size for debugging
    disp(['Grid size: ', num2str(size(grid,1)), 'x', num2str(size(grid,2))]);
    disp(['Grid class: ', class(grid)]);
    
    % Perform quadtree decomposition
    qt = quadtreeDecomposition(grid);
    
    % Display the original grid
    figure('Name', 'Quadtree Decomposition Test', 'NumberTitle', 'off');
    
    subplot(1,2,1);
    imagesc(grid);
    colormap(gray);
    axis equal tight off;
    title('Original Map with Four Obstacles');
    
    % Display the quadtree decomposition
    subplot(1,2,2);
    imagesc(grid);
    colormap(gray);
    hold on;
    axis equal tight off;
    title('Quadtree Decomposition');
    
    % Traverse the quadtree and plot rectangles for leaf nodes
    traverseAndPlot(qt, gridSize);
    
    hold off;
end

function traverseAndPlot(node, gridSize)
% Traverse the quadtree and plot rectangles for leaf nodes

    if isempty(node.children)
        % Leaf node: plot rectangle
        x = node.x;
        y = node.y;
        size = node.size;
        
        % Adjust for MATLAB's 1-based indexing and plotting
        % rectangle Position: [x, y, width, height]
        % MATLAB's image y-axis is top to bottom, so no need to flip y
        
        rectangle('Position', [x-0.5, y-0.5, size, size], ...
                  'EdgeColor', 'r', 'LineWidth', 1);
    else
        % Internal node: traverse children
        for i = 1:length(node.children)
            traverseAndPlot(node.children(i), gridSize);
        end
    end
end