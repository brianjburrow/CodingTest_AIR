import pandas
import numpy as np 
import os, sys
from glob import iglob
import csv
import time
class CatalogProcessor:
    def __init__(self, num_catalog_years, file_prefix, num_files):
        ## CatalogProcessor constructor
        # Input: integer: num_catalog_years - number of years contained in the stochastic catalog
        #        string: file_prefix - file names are assumed to be of the form "file_prefix_%d.csv", where %d is
        #                              the part number.  E.g.,  10k_catalog_losses_part_1.csv, 10k_catalog_losses_part_2.csv, etc.
        #                              In this example, 10k_catalog_losses_part_ is the file prefix.  .csv is the suffix. 
        #       integer: num_files - The number of parts to the dataset.  In the above example, we have shown two parts.  numbering should start at 1.
        # Output: pandas.DataFrame stored at "./outfiles/year_loss_dataframe_%d.csv"%self.num_catalog_years"
        #         This data frame contains the "Loss" entries for each county in the dataset, for all years in the dataset.  The headers are the cresta codes for each county
        #         This file in the input file for computing the Average Annual Loss (AAL), Standard Deviation (STD), Standard Error (STD_ERR), and Coefficient of Variation (COV)
        #         That will be used for assessing catalog convergence.
        if not os.path.exists("./outfiles/"):
            os.mkdir("./outfiles/")
            os.mkdir("./outfiles/aal_tables/")
            os.mkdir("./outfiles/aal_tables/counties/")
            os.mkdir("./outfiles/aal_tables/states/")

        ## define hard-coded values
        self.convergence_criterion = 0.05 # A.I.R's standard convergence criterion is a coefficient of variation less than 5%, or 0.05
        self.file_suffix           = ".csv"

        self.outfile_name_prefix   = "outfiles/year_loss_table_%dk_cresta_code_"%num_catalog_years

        ## define loaded in parameters
        self.num_catalog_years     = num_catalog_years
        self.file_prefix           = file_prefix 
        self.num_files             = num_files 

        ## check input for correctness
        self._check_num_catalog_years()

        ## Initialize dataframe for year_loss_table
        self.year_loss_table         = None 
        self.unique_cresta_codes     = None
        self.cresta_strings          = []
        self.year_loss_dataframe_set = []

        self._initialize_year_loss_table()

        self.compute_year_loss_table()
        pass


    def _check_num_catalog_years(self):
        # Check to ensure that the number of catalog years is supported by CatalogProcessor.
        # Currently, 10k or 100k are accepted.
        if(self.num_catalog_years == 10000 or self.num_catalog_years == 100000):
            pass
        else:
            raise Exception("CatalogProcessor: num_catalog_years must be either 10000, or 100000.")
        pass
    
    def compute_year_loss_table(self):
        print("Getting unique cresta codes")
        self.unique_cresta_codes = []
        for iDataset in np.arange(1, self.num_files + 1):
            self.unique_cresta_codes += self._get_cresta_codes(iDataset)
            self.unique_cresta_codes = np.unique(self.unique_cresta_codes).tolist()  # this can get out of hand without reduction if the number of files is large, so we reduce at each loop
        self.unique_cresta_codes = np.unique(self.unique_cresta_codes)
        print("Done Getting unique cresta codes")
        for iDataset in np.arange(1, self.num_files + 1):
            self._process_single_dataset(iDataset)
        self._convert_loss_matrix_to_data_frame_and_save()
        pass

    def _get_cresta_codes(self, part_number):
        filename  = "%s%d%s"%(self.file_prefix, part_number, self.file_suffix)
        temp_data = pandas.read_csv(filename, usecols = ["CrestaCode"], dtype = {"CrestaCode": 'str'}, low_memory = False)
        return temp_data["CrestaCode"].unique().tolist()




    def _process_single_dataset(self, part_number):
        filename  = "%s%d%s"%(self.file_prefix, part_number, self.file_suffix)

        print("Processing %s"%filename)

        temp_data = pandas.read_csv(filename, usecols = ["YearID", "CrestaCode", "GrossLoss"], dtype = {"YearID": 'int', "CrestaCode": 'str', "GrossLoss": 'float'}, low_memory = False)

        td_2 = pandas.pivot_table(temp_data, values = "GrossLoss", index = "YearID", columns = "CrestaCode", aggfunc = np.sum)
        
        td_2 = td_2.fillna(0)

        indices     = np.sort(temp_data["YearID"].unique())

        cresta_nums = list(td_2.columns.values)
        
        matrix = td_2.to_numpy()                                     # never do this if a dataframe has mixed data types in the columns, it will be super slow.  See Pandas documentation

        self.year_loss_dataframe_set.append(pandas.DataFrame(data = matrix, index = indices, columns = cresta_nums)) # create a nicely organized dataframe and append to our year_loss_dataframe_set.  We will merge later.
        pass

    def _convert_loss_matrix_to_data_frame_and_save(self):
        nDataFrames = len(self.year_loss_dataframe_set)
        start = pandas.concat([self.year_loss_dataframe_set[0], self.year_loss_dataframe_set[1]])
        for iDataFrame in range(2,nDataFrames):
            start = pandas.concat([start, self.year_loss_dataframe_set[iDataFrame]])

        print(start)
        start = start.fillna(0)
        start.to_csv("./outfiles/year_loss_dataframe_%d.csv"%self.num_catalog_years, index = False)
        pass
 
    def _initialize_year_loss_table(self):
        self.year_loss_table = pandas.DataFrame(data = None, columns = ["Year", "Loss"], index = np.arange(1, self.num_catalog_years + 1))
        self.year_loss_table.loc[:, "Year"] = np.arange(1, self.num_catalog_years + 1)
        pass
# End class definition for CatalogConstructor


def compute_requirements(dataset, cresta_string, num_years, location_type = "states"):
    # This function computes the AAL, STD, STD_ERR, and COV from a particular dataset
    # Input:  dataset : pandas.DataFrame : one dimensional container with aggregate loss data associated with each calendar year in the catalog
    # Input: cresta_string : Either a 2 or 5 character string containing either the state i.d. if location_type == "states", or the state and county i.d. if the location_type == "counties"
    # Input: num_years : integer : number of years in the stochastic catalog
    # Input: location_type : string : either "states" or "counties", depending on how dataset was preprocessed prior to this function call.
                                                          # must convert to use np.std() with the ddof = 1 option.
    print("Computing AAL Tables for %s"%cresta_string)
    nData = len(dataset)
    output = np.zeros([nData, 4]) # 4 columns, AAL, STD, STD_ERR, COV respectively

    output[:, 0] = dataset.expanding(1).mean()                    # pandas DataFrame.expanding(num).mean() computes the cumulative mean
    output[:, 1] = dataset.expanding(1).std()                     # pandas DataFrame.expanding(num).std() computes the cumulative standard devation with 1/(n - 1) divisor
    output[1:, 2] = output[1:,1]/np.sqrt(np.arange(2, nData+1))
    output[1:, 3] = output[1:,2]/output[1:,0]

    dataframe = pandas.DataFrame(data = output, columns = ["AAL", "STD", "STD_ERR", "COV"])

    dataframe.to_csv("outfiles/aal_tables/%s/%s_aal_tab_%d.csv"%(location_type, cresta_string, num_years), index = False)
    pass 

def process_by_state(num_years):
    # Input: num_years   : Type integer : the number of years contained in the stochastic catalog
    # Output: a series of .csv files containing the AAL, STD, STD_ERR, COV for losses in each state.

    state_prefix_set = get_unique_state_prefixes(num_years)
    dataset        = pandas.read_csv("./outfiles/year_loss_dataframe_%d.csv"%num_years, low_memory = False)

    for state_prefix in state_prefix_set:
        filter_col     = [col for col in dataset if col.startswith(state_prefix)]

        trim_dataset   = dataset[filter_col]

        loss_by_state  = trim_dataset.sum(axis = 1)                                                 # sum the columns to obtain the aggregate year loss per state rather than by county

        outfile_prefix = "by_state_cresta_prefix_%s"%state_prefix

        compute_requirements(loss_by_state, outfile_prefix, num_years, location_type = "states")
    pass


def process_by_county(num_years):
    # Input: num_yeras   : Type integer : the number of years contained in the stochastic catalog
    # Output: a series of .csv files containing the AAL, STD, STD_ERR, COV for losses in each county contained in the dataset.
    # The files are stored by CRESTA identifier (a string with 5 characters denoting the state and county)
    # The unique cresta identifiers are inferred from year_loss_dataframe_%d.csv%num_years.
    dataset        = pandas.read_csv("./outfiles/year_loss_dataframe_%d.csv"%num_years, low_memory = False)

    cresta_strings = dataset.columns.values.tolist()

    nData = len(dataset)

    for cresta_string in cresta_strings:

        partial_data  = dataset[cresta_string]

        compute_requirements(partial_data, cresta_string, num_years, location_type = "counties")
    pass    

def get_unique_state_prefixes(num_years):
    # Input: num_years   : Type integer : the number of years contained in the stochastic catalog
    # Output: state_prefixes : type set: contains a list of unique CRESTA identifiers for each state.  Each identifier should be a string with two numbers. e.g., 01, 07, 11, 47, etc.
    filename = "./outfiles/year_loss_dataframe_%d.csv"%num_years
    unique_headers = pandas.read_csv(filename, nrows=1).columns.tolist()

    state_prefixes = set()
    for county in unique_headers:
        state_prefixes.add(county[0:2])
    print(state_prefixes)
    return(state_prefixes)


if __name__ == "__main__": 
    num_cat_years = 100000                         # 100000 year catalog
    prefix        = "./100K/Unfiltered_100k_20190919_part_"
    nFiles        = 100

    # num_cat_years = 10000                         # 10000 year year catalog
    # prefix        = "./10K/10k_20180312_part_"    # folder and file prefixes for the individual parts of the chunked loss files.
    # nFiles        = 3

    # Preprocess Catalog and save reduced file
    cp            = CatalogProcessor(num_cat_years, prefix, nFiles)  # this line can be commented out once it has been run for a particular catalog once.  Can save time significantly if you want to change the next two functions.
    ## perform analysis 
    process_by_county(num_cat_years)
    process_by_state(num_cat_years)