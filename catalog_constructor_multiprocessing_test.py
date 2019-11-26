import pandas
import numpy as np 
import os, sys
from glob import iglob
import csv
from multiprocessing import Pool
import multiprocessing

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
        self.year_loss_table       = None 
        self.unique_cresta_codes   = None
        self.cresta_strings        = []

        self._initialize_year_loss_table()
        pass


    def _check_num_catalog_years(self):
        # Check to ensure that the number of catalog years is supported by CatalogProcessor.
        # Currently, 10k or 100k are accepted.
        if(self.num_catalog_years == 10000 or self.num_catalog_years == 100000):
            pass
        else:
            raise Exception("CatalogProcessor: num_catalog_years must be either 10000, or 100000.")
        pass
    
    def compute_year_loss_table(self, iDataset):
        return(self._process_single_dataset(iDataset))


    def _process_single_dataset(self, part_number):
        filename  = "%s%d%s"%(self.file_prefix, part_number, self.file_suffix)

        print("Processing %s"%filename)

        temp_data = pandas.read_csv(filename, usecols = ["YearID", "CrestaCode", "GrossLoss"], low_memory = False)


        # Makes the assumption that all crestacodes that exist in the dataset are contained in the first file.  
        # This may not be true in general, but it seems like a reasonable assumption.  I will put a check in later.
        self.unique_cresta_codes = temp_data.loc[:,"CrestaCode"].unique()
        # if you don't get a memory error initializing the next line, then the code should 
        # be able to complete without issue.
        year_loss_matrix    = np.zeros([len(temp_data), len(self.unique_cresta_codes)])
        
        unique_years = temp_data["YearID"].unique()

        counter = 0
        for year_num in unique_years:
            # losses exist for that year, find the cresta codes that also have data for that year
            # if losses do not exist for that year, the entry was already populated with zero loss 
            # during initialization of self.year_loss_matrix
            filtered_df = temp_data.loc[temp_data["YearID"] == year_num, :]
            cresta_set  = filtered_df["CrestaCode"].unique()

            for iCode, cresta_code in enumerate(self.unique_cresta_codes):
                if cresta_code in cresta_set:
                    # if the current cresta_code has data for the current year_num, then fill the table
                    # otherwise it is already set to zero during initialization of self.year_loss_matrix
                    year_loss_matrix[counter, iCode] = filtered_df.loc[filtered_df["CrestaCode"] == cresta_code, :]["GrossLoss"].sum()
            counter += 1
        return(year_loss_matrix)

    def convert_loss_matrix_to_data_frame_and_save(self, year_loss_matrix):
        for cresta_code in self.unique_cresta_codes:
            cresta_as_string = "{}".format(cresta_code)                 # convert from int to string
            if(len(cresta_as_string) == 4):
                # add leading zero back
                cresta_as_string = "0" + cresta_as_string
            self.cresta_strings.append(cresta_as_string)

        self.year_loss_dataframe = pandas.DataFrame(data = year_loss_matrix, \
            index = np.arange(1, year_loss_matrix.shape[0] + 1), \
            columns = self.cresta_strings)

        self.year_loss_matrix = None # clear the memory
        self.year_loss_dataframe.to_csv("./outfiles/year_loss_dataframe_%d.csv"%self.num_catalog_years, index = False)
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

    dataset = dataset.to_numpy()                                                           # must convert to use np.std() with the ddof = 1 option.
    print("Computing AAL Tables for %s"%cresta_string)
    nData = len(dataset)
    output = np.zeros([nData, 4])                                                          # 4 columns, AAL, STD, STD_ERR, COV respectively

    output[:, 0] = dataset.cumsum()/np.arange(1, nData+1)

    for iData in np.arange(1, nData):
        output[iData, 1] = np.std(dataset[0:(iData + 1)] , dtype  = np.float64, ddof = 1)  # ddof makes sure that we are using the sample standard deviation
    
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

        loss_by_state  = trim_dataset.sum(axis = 1)                                                 # sum the columns to obtain the year loss per state rather than by county

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
    num_cat_years = 100000                         # 10000 year or 100000 year catalog
    #prefix        = "./10K/10k_20180312_part_"    # folder and file prefixes for the individual parts of the chunked loss files.
    prefix        = "./100K/Unfiltered_100k_20190919_part_"
    nFiles        = 100

    # Preprocess Catalog and save reduced file
    cp            = CatalogProcessor(num_cat_years, prefix, nFiles)  # this line can be commented out once it has been run for a particular catalog once.  Can save time significantly if you want to change the next two functions.
    year_loss_tables = []*nFiles

    p = Pool(processes = min(nFiles, (multiprocessing.cpu_count() - 1) ))
    year_loss_matrix = p.map(cp.compute_year_loss_table, [iDataset for iDataset in np.arange(1, nFiles + 1)])
    cp.convert_loss_matrix_to_data_frame_and_save(year_loss_matrix)
    ## perform analysis 
    process_by_county(num_cat_years)
    process_by_state(num_cat_years)