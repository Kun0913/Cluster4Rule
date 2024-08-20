import os

def process_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    processed_lines = []
    for line in lines:
        columns = line.split()
        if len(columns) >= 16:  # 确保有足够的列
            new_line = columns[:15] + columns[16:]  # 删除第8、9列
            processed_lines.append(' '.join(new_line) + '\n')
        else:
            processed_lines.append(line)

    with open(file_path, 'w') as file:
        file.writelines(processed_lines)

def process_specified_txt_file(file_name):
    # 获取当前脚本文件所在目录的绝对路径
    current_dir = os.getcwd()
    # 构造 "data" 目录的路径
    data_dir = os.path.join(current_dir, "../data")  # 返回上一级目录下的 "data" 目录
    # 定义文件名
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path):
        process_txt_file(file_path)
        print(f"Processed {file_path}")
    else:
        print(f"The file {file_path} does not exist.")

if __name__ == "__main__":
    file_name = f"output_state_2vs2.txt"  # 指定文件名
    process_specified_txt_file(file_name)
