from tqdm import tqdm
def extract_boundaries(filename):
    with open(filename, 'r') as file:
        # 读取第一行，初始化上下界数组
        first_line = file.readline().strip().split()
        num_dimensions = len(first_line)
        upper_boundaries = [float('-inf')] * num_dimensions
        lower_boundaries = [float('inf')] * num_dimensions

        # 遍历每行数据，更新上下界数组
        for line in tqdm(file,desc='boundary'):
            data_point = line.strip().split()
            for i in range(num_dimensions):
                value = float(data_point[i])
                upper_boundaries[i] = max(upper_boundaries[i], value)
                lower_boundaries[i] = min(lower_boundaries[i], value)

    return upper_boundaries, lower_boundaries