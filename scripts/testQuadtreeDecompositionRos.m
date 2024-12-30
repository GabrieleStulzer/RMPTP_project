function testQuadtreeDecompositionRos()
% TESTQUADTREEDECOMPOSITION Test script for quadtree decomposition with four rectangular obstacles.

    % Clear workspace and figures to avoid conflicts
    clear all
    close all
    clc

    rosinit;

    node = ros2node('/shelfino1/global_costmap/global_costmap');

    subscriber = ros2subscriber(node, '/shelfino1/global_costmap/costmap');

    % amcl_node = ros2node('/shelfino1/amcl');
    % amcl_pose = ros2subscriber(amcl_node, '/shelfino1/amcl_pose');

    msg = receive(subscriber);

    grid = reshape(msg.data, msg.info.width, msg.info.height);
    grid = ~grid;
    
    % Convert grid to double (if not already)
    grid = double(grid);

    [rows, cols] = size(grid);
    % Ensure the grid is square and size is a power of 2 for simplicity
    % maxSize = max(rows, cols);
    % power = nextpow2(maxSize);
    % newSize = 2^power;
    % paddedGrid = zeros(newSize);
    % paddedGrid(1:rows, 1:cols) = grid;
    % grid = paddedGrid;
    
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
    traverseAndPlot(qt, cols);
    
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