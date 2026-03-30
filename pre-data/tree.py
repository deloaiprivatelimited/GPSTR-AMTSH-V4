import os

def print_tree(start_path='.', output_file='folder_tree.txt'):
    with open(output_file, 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(start_path):
            
            # ✅ Skip virtual environment folders
            dirs[:] = [d for d in dirs if d not in ('venv', '.venv', 'env', '__pycache__','.git','textbooks','master_data','merged','validation_results','validation_reports')]

            level = root.replace(start_path, '').count(os.sep)
            indent = '│   ' * level
            folder_name = os.path.basename(root) if os.path.basename(root) else root
            
            line = f"{indent}├── {folder_name}/"
            print(line)
            f.write(line + '\n')

            sub_indent = '│   ' * (level + 1)
            for file in files:
                file_line = f"{sub_indent}├── {file}"
                print(file_line)
                f.write(file_line + '\n')

if __name__ == "__main__":
    print_tree()