import sys
from AAAI_paper.create_behavior_policy import evaluate_all_policies

if __name__ == "__main__":
    filename = sys.argv[1]

    if filename[-1] == "/":
        evaluate_all_policies(foldername=filename)
    elif filename[-3:] == 'csv':
        evaluate_all_policies(filename=filename)
    else:
        print("Provide a foldername that ends with '/' or provide a filename that ends with .csv")
