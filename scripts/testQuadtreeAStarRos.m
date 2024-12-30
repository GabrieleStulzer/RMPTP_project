function testQuadtreeAStarRos()
% TESTQUADTREEASTAR Test script for quadtree decomposition and A* pathfinding.

    % Clear workspace and figures to avoid conflicts
    clear all
    close all
    clc

    %% 1. Initialize and Load map from ros topic

    rosinit;

    node = ros2node('/shelfino1/global_costmap/global_costmap');

    subscriber = ros2subscriber(node, '/shelfino1/global_costmap/costmap');

    % amcl_node = ros2node('/shelfino1/amcl');
    % amcl_pose = ros2subscriber(amcl_node, '/shelfino1/amcl_pose');

    msg = receive(subscriber);

    grid = reshape(msg.data, msg.info.width, msg.info.height);
    

    % Convert grid to double (if not already)
    grid = double(grid);
    
    % Display grid type and size for debugging
    disp(['Grid size: ', num2str(size(grid,1)), 'x', num2str(size(grid,2))]);
    disp(['Grid class: ', class(grid)]);
    
    %% 2. Perform Quadtree Decomposition
    
    qt = quadtreeDecomposition(grid);
    
    %% 3. Extract Free Leaf Nodes from Quadtree
    
    leafNodes = extractFreeLeafNodes(qt);
    disp(['Number of free leaf nodes: ', num2str(length(leafNodes))]);
    
    %% 4. Assign Unique IDs to Free Leaf Nodes
    
    % Assign unique IDs to free leaf nodes
    for i = 1:length(leafNodes)
        leafNodes(i).id = i;
    end
    
    % Debugging: Verify that 'id's are assigned
    if all(arrayfun(@(n) isfield(n, 'id'), leafNodes))
        disp('ID field successfully assigned to all free leaf nodes.');
    else
        error('Failed to assign ID fields to some free leaf nodes.');
    end
    
    %% 5. Define Start and Goal Positions
    
    % Define start and goal positions (ensure they are in free space)
    startPos = [1000, 2000];  % [x, y]
    goalPos = [3000, 2500]; % [x, y]
    
    % Find the free leaf nodes containing the start and goal positions
    startNode = findContainingNode(leafNodes, startPos);
    goalNode = findContainingNode(leafNodes, goalPos);
    
    if isempty(startNode) || isempty(goalNode)
        error('Start or Goal position is inside an obstacle or outside the grid.');
    end
    
    disp(['Start Node ID: ', num2str(startNode.id), ' at (', num2str(startNode.x), ',', num2str(startNode.y), ') Size: ', num2str(startNode.size)]);
    disp(['Goal Node ID: ', num2str(goalNode.id), ' at (', num2str(goalNode.x), ',', num2str(goalNode.y), ') Size: ', num2str(goalNode.size)]);
    
    %% 6. Build Graph from Free Leaf Nodes
    
    % Build adjacency list
    adjacencyList = buildAdjacencyList(leafNodes, gridSize);
    
    % Debugging: Check sample adjacency entries
    disp('Sample adjacency list entries:');
    for i = 1:min(5, length(adjacencyList))
        disp(['Node ', num2str(i), ' neighbors: ', num2str(adjacencyList{i}')]);
    end
    
    %% 7. Implement A* Algorithm to Find Path
    
    % Perform A* search
    path = aStar(startNode, goalNode, leafNodes, adjacencyList);
    
    if isempty(path)
        disp('No path found from Start to Goal.');
    else
        disp(['Path found with ', num2str(length(path)), ' nodes.']);
    end
    
    %% 8. Visualization
    
    % Display the original grid and quadtree decomposition
    figure('Name', 'Quadtree Decomposition and A* Pathfinding', 'NumberTitle', 'off');
    
    % Subplot 1: Original Map
    subplot(1,2,1);
    imagesc(grid);
    colormap(gray);
    axis equal tight off;
    title('Original Map with Four Obstacles');
    hold on;
    plot(startPos(1), startPos(2), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g'); % Start
    plot(goalPos(1), goalPos(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');   % Goal
    hold off;
    
    % Subplot 2: Quadtree Decomposition with Path
    subplot(1,2,2);
    imagesc(grid);
    colormap(gray);
    hold on;
    axis equal tight off;
    title('Quadtree Decomposition with A* Path');
    plot(startPos(1), startPos(2), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g'); % Start
    plot(goalPos(1), goalPos(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');   % Goal
    
    % Traverse the quadtree and plot rectangles for free leaf nodes
    traverseAndPlot(qt, gridSize);
    
    % Overlay the path if found
    if ~isempty(path)
        pathPoints = [];
        for i = 1:length(path)
            node = leafNodes(path(i));
            centerX = node.x + node.size / 2;
            centerY = node.y + node.size / 2;
            pathPoints = [pathPoints; centerX, centerY];
        end
        plot(pathPoints(:,1), pathPoints(:,2), 'b-', 'LineWidth', 2);
        plot(pathPoints(:,1), pathPoints(:,2), 'b*', 'MarkerSize', 5);
    end
    
    hold off;
end

%% Helper Functions

function leafNodes = extractFreeLeafNodes(qt)
% EXTRACTFREELEAFNODES Traverse the quadtree and extract all free leaf nodes.
%
% Only nodes with value == 0 (free space) are extracted.

    leafNodes = [];
    stack = qt; % Initialize stack with root node
    
    while ~isempty(stack)
        current = stack(1);
        stack(1) = []; % Pop the first element
        
        if isempty(current.children)
            if current.value == 0
                leafNodes = [leafNodes, current];
            end
        else
            stack = [stack, current.children];
        end
    end
end

function node = findContainingNode(leafNodes, pos)
% FINDCONTAININGNODE Find the free leaf node that contains the given position.
% pos: [x, y]

    x = pos(1);
    y = pos(2);
    
    for i = 1:length(leafNodes)
        current = leafNodes(i);
        if x >= current.x && x < (current.x + current.size) && ...
           y >= current.y && y < (current.y + current.size)
            node = leafNodes(i);
            return;
        end
    end
    node = []; % Not found or inside an obstacle
end

function adjacencyList = buildAdjacencyList(leafNodes, gridSize)
% BUILDADJACENCYLIST Build adjacency list for all free leaf nodes based on spatial adjacency.

    numNodes = length(leafNodes);
    adjacencyList = cell(numNodes, 1);
    
    % Create a spatial index for quick lookup
    spatialIndex = containers.Map('KeyType', 'char', 'ValueType', 'int32');
    for i = 1:numNodes
        node = leafNodes(i);
        key = sprintf('%d_%d', node.x, node.y);
        spatialIndex(key) = i;
    end
    
    % Directions: N, S, E, W, NE, NW, SE, SW
    directions = [
        0, -1;  % N
        0, 1;   % S
        1, 0;   % E
        -1, 0;  % W
        1, -1;  % NE
        -1, -1; % NW
        1, 1;   % SE
        -1, 1    % SW
    ];
    
    for i = 1:numNodes
        node = leafNodes(i);
        neighbors = [];
        
        % Calculate potential neighbor top-left coordinates
        for d = 1:size(directions,1)
            dx = directions(d,1) * node.size;
            dy = directions(d,2) * node.size;
            
            neighborX = node.x + dx;
            neighborY = node.y + dy;
            
            % Ensure neighbor coordinates are within grid bounds
            if neighborX < 1 || neighborY < 1 || neighborX > gridSize || neighborY > gridSize
                continue;
            end
            
            % Find the free leaf node that contains (neighborX, neighborY)
            key = sprintf('%d_%d', neighborX, neighborY);
            if spatialIndex.isKey(key)
                neighborIdx = spatialIndex(key);
                neighbors = [neighbors; neighborIdx];
            end
        end
        
        adjacencyList{i} = unique(neighbors); % Remove duplicate neighbors
    end
end

function path = aStar(startNode, goalNode, leafNodes, adjacencyList)
% ASTAR Perform A* search from startNode to goalNode on the graph.

    openSet = [];
    closedSet = false(length(leafNodes),1);
    gScore = Inf(length(leafNodes),1);
    fScore = Inf(length(leafNodes),1);
    cameFrom = zeros(length(leafNodes),1);
    
    startIdx = startNode.id;
    goalIdx = goalNode.id;
    
    gScore(startIdx) = 0;
    fScore(startIdx) = heuristic(leafNodes(startIdx), leafNodes(goalIdx));
    
    openSet = [openSet; startIdx];
    
    while ~isempty(openSet)
        % Find the node in openSet with the lowest fScore
        [~, minIdx] = min(fScore(openSet));
        current = openSet(minIdx);
        
        if current == goalIdx
            % Reconstruct path
            path = reconstructPath(cameFrom, current);
            return;
        end
        
        % Remove current from openSet
        openSet(minIdx) = [];
        closedSet(current) = true;
        
        % Iterate through neighbors
        neighbors = adjacencyList{current};
        for i = 1:length(neighbors)
            neighbor = neighbors(i);
            if closedSet(neighbor)
                continue;
            end
            
            tentative_gScore = gScore(current) + distance(leafNodes(current), leafNodes(neighbor));
            
            if ~ismember(neighbor, openSet)
                openSet = [openSet; neighbor];
            elseif tentative_gScore >= gScore(neighbor)
                continue;
            end
            
            % This path is the best until now
            cameFrom(neighbor) = current;
            gScore(neighbor) = tentative_gScore;
            fScore(neighbor) = gScore(neighbor) + heuristic(leafNodes(neighbor), leafNodes(goalIdx));
        end
    end
    
    % If we reach here, no path was found
    path = [];
end

function h = heuristic(node, goalNode)
% HEURISTIC Compute the heuristic (Euclidean distance) between node and goalNode.

    nodeCenter = [node.x + node.size / 2, node.y + node.size / 2];
    goalCenter = [goalNode.x + goalNode.size / 2, goalNode.y + goalNode.size / 2];
    h = norm(nodeCenter - goalCenter);
end

function d = distance(nodeA, nodeB)
% DISTANCE Compute the distance between nodeA and nodeB (Euclidean).

    centerA = [nodeA.x + nodeA.size / 2, nodeA.y + nodeA.size / 2];
    centerB = [nodeB.x + nodeB.size / 2, nodeB.y + nodeB.size / 2];
    d = norm(centerA - centerB);
end

function path = reconstructPath(cameFrom, current)
% RECONSTRUCTPATH Reconstruct the path from start to goal.

    path = [current];
    while cameFrom(current) ~= 0
        current = cameFrom(current);
        path = [current, path];
    end
end

function traverseAndPlot(node, gridSize)
% TRAVERSEANDPLOT Traverse the quadtree and plot rectangles for free leaf nodes.

    if isempty(node.children)
        % Leaf node: plot rectangle
        x = node.x;
        y = node.y;
        size = node.size;
        
        % Adjust for MATLAB's 1-based indexing and plotting
        rectangle('Position', [x-0.5, y-0.5, size, size], ...
                  'EdgeColor', 'r', 'LineWidth', 0.5);
    else
        % Internal node: traverse children
        for i = 1:length(node.children)
            traverseAndPlot(node.children(i), gridSize);
        end
    end
end