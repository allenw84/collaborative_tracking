def cal_ratio(box_range, grid_size, grid_pos):
    x_min = grid_size[0] * grid_pos[0]
    x_max = grid_size[0] * (grid_pos[0] + 1)
    y_min = grid_size[1] * grid_pos[1]
    y_max = grid_size[1] * (grid_pos[1] + 1)

    x_leftblank = box_range[0][0] - x_min
    x_rightblank = x_max - box_range[1][0]
    y_topblank = box_range[0][1] - y_min
    y_bottomblank = y_max - box_range[1][1]

    if x_leftblank < 0: x_leftblank = 0
    if x_rightblank < 0: x_rightblank = 0
    if y_topblank < 0: y_topblank = 0
    if y_bottomblank < 0: y_bottomblank = 0

    x_blank = x_leftblank + x_rightblank
    y_blank = y_topblank + y_bottomblank

    return (1 - x_blank / grid_size[0]) * (1 - y_blank / grid_size[1])
    
def find_grids(box, grid_size,  min_ratio):
    x_minPos = int(box[0] / grid_size[0])
    y_minPos = int(box[1] / grid_size[1])
    x_maxPos = int(box[2] / grid_size[0])
    y_maxPos = int(box[3] / grid_size[1])

    ans = []

    for i in range(x_minPos, x_maxPos + 1):
        for j in range(y_minPos, y_maxPos + 1):
            if cal_ratio([[box[0], box[1]], [box[2], box[3]]], grid_size, [i, j]) > min_ratio:
                ans.append([i, j])

    return ans
    
    
    
#function calling example
if __name__=="__main__":
    find_grids(output, [x_grid, y_grid], 0.1)
    #output is the output box from detector
    #[x_grid, y_grid] is the grid size for x-axis and y-axis
    #0.1 is the threshold for box size in the grid