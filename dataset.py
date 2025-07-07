import kagglehub

# Download latest version
path = kagglehub.dataset_download("nirmalsankalana/apple-tree-leaf-disease-dataset")

print("Path to dataset files:", path)