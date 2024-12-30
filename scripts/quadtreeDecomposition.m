function qt = quadtreeDecomposition(grid)
% QUADTREEDECOMPOSITION Perform quadtree decomposition on a binary occupancy grid.
%
%   qt = QUADTREEDEcomposition(grid) takes a binary occupancy grid (2D matrix)
%   as input and returns a quadtree structure where each leaf node satisfies
%   one of the following conditions:
%       - 100% of the cells are 0
%       - At least 80% of the cells are 1
%
%   The quadtree structure has the following fields:
%       - x: x-coordinate of the top-left corner
%       - y: y-coordinate of the top-left corner
%       - size: size of the current cell
%       - value: 0 or 1 if it's a leaf node, empty otherwise
%       - children: array of 4 child nodes if not a leaf, empty otherwise

    % Validate input
    if ~ismatrix(grid) || ~(isnumeric(grid) || islogical(grid))
        error('Input must be a 2D numeric or logical matrix.');
    end
    
    % Convert logical to double for consistent processing
    if islogical(grid)
        grid = double(grid);
    end

    [rows, cols] = size(grid);
    % Ensure the grid is square and size is a power of 2 for simplicity
    maxSize = max(rows, cols);
    power = nextpow2(maxSize);
    newSize = 2^power;
    paddedGrid = zeros(newSize);
    paddedGrid(1:rows, 1:cols) = grid;
    
    qt = buildQuadtree(paddedGrid, 1, 1, newSize);
end

function node = buildQuadtree(grid, x, y, size)
% Recursive function to build quadtree

    node = struct('x', x, 'y', y, 'size', size, 'value', [], 'children', []);
    
    currentRegion = grid(y:y+size-1, x:x+size-1);
    totalCells = numel(currentRegion);
    numOnes = sum(currentRegion(:));
    
    % Check leaf conditions
    if numOnes == 0
        node.value = 0;
        return;
    elseif (numOnes / totalCells) >= 0.8
        node.value = 1;
        return;
    end
    
    % If size is 1, make it a leaf node
    if size == 1
        node.value = currentRegion;
        return;
    end
    
    % Subdivide into four quadrants
    halfSize = floor(size / 2);
    if halfSize == 0
        halfSize = 1; % Minimum size of 1
    end
    
    children = [];
    offsets = [0 0; halfSize 0; 0 halfSize; halfSize halfSize];
    for i = 1:4
        newX = x + offsets(i,1);
        newY = y + offsets(i,2);
        child = buildQuadtree(grid, newX, newY, halfSize);
        children = [children, child];
    end
    node.children = children;
end