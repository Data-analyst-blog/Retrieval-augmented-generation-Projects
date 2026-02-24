import os

def load_files(data_path="data"):
    documents = []

    for dept in os.listdir(data_path):
        dept_path = os.path.join(data_path, dept)

        if os.path.isdir(dept_path):
            for file in os.listdir(dept_path):
                file_path = os.path.join(dept_path, file)

                documents.append({
                    "file_path": file_path,
                    "department": dept,
                    "file_name": file
                })

    return documents