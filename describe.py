from src.preprocessing.file.loader import Loader

def main():
    dataset = Loader().load("datasets/dataset_train.csv")

    headers = list(dataset.get_headers().keys())
    
    numerics_headers = headers[dataset.numeric_data_index:]
    numerics = dataset.get_numerics()

    for i, header in enumerate(numerics_headers):
        print()
    print(numerics_headers)
if __name__ == "__main__":
    main()