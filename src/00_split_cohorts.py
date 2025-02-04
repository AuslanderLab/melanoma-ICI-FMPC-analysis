import pandas as pd

"""
    This script provides the quick splitting of the supplementary dataset 4 into its training cohorts using pandas.
"""

if __name__ == "__main__":
    dataset_4 = pd.read_csv("data/supp_dataset_4.csv")
    cln = pd.read_csv('data/supp_dataset_1.csv',
                      index_col='Sample', na_values=["N_A", "NA", ""])
    
    # Get indices/columns by study
    dallas_ids = cln[cln['Study'] == 'Dallas'].index
    pittsburgh_ids = cln[cln['Study'] == 'Pittsburgh'].index
    ny_ids = cln[cln['Study'] == 'new_york'].index
    # Create paired dataframes
    dallas_expression = dataset_4[dallas_ids]
    pittsburgh_expression = dataset_4[pittsburgh_ids]
    ny_expression = dataset_4[ny_ids]
    dallas_expression.to_csv("results/supp_dataset_4-DALLAS.csv")
    pittsburgh_expression.to_csv("results/supp_dataset_4-PITT.csv")
    ny_expression.to_csv("results/supp_dataset_4-NY.csv")
