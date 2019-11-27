./Python_Test


Computing Resource Requirements

- Hardware Requirements
-- 12 GB of storage space for storing 10K and 100K Loss Files (if you store the files locally)
-- ~8 GB of storage space for temporary files.
-- 8-20 GB total 

- Software Requirements (earlier versions may work, but no guarantees)
-- Windows 10 64 Bit Operating System (parts tested on Mac with no issues.  But, no guarantees on the final product)
-- Python3 (may work on Python2, but not tested)
-- Python3 Packages
---- pandas 
---- numpy
---- csv
---- matplotlib

-- Python3 packages that should come with your Python3 distribution (at least they came with mine)
---- glob

- Computing Time
--- This pipeline is somewhat computationally expensive.  ~ 10 minutes to process the 100K catalog, and ~6 minutes for visualizing the 100K results.  
--- Not all of the plots are necessary, and can be commented out to reduce the computational time (getting the AAL plots with error for each county is expensive.)

--- Part of the computational cost is due to me coding up a functional pipeline
--- without using Pandas to its full extent in the beginning of the code, thus I did not format the latter parts to accept
--- fancy pandas dataframes. Rather than rewrite the pipeline, I have decided to live with a somewhat reasonably fast code.
--- Numpy is great for a lot of things, but dealing with complicated data matrices (i.e., DataFrames that need level based analysis)
--- is not something it should be used for.  For loops on pandas Dataframes are terribly slow, and vectorizing level based analysis
--- cannot be done easily without using built in Pandas functions.
 
--- Also the way that I set up the temporary files is not ideal, and could be changed in later versions to plot the output
--- on the fly, rather than saving temporary files.  This would be necessary if the catalogs got much larger.  Granted,
--- it is nice to have the temporary files for smaller datasets for additional post processing if desired.

--- I did not consider that part_1 might not contain all of the cresta codes, which was not a good assumption.  I had to
--- rework a few things in a bit of an add hoc way (see _get_cresta_codes() implementation).  This results in unnecessary slow
--- down for the large catalogs.

--- That being said, the code is down from a few hours for the 100K library to ~ 10 minutes, but there is still a lot of 
--- performance to be gained by addressing the above issues.  Unfortunately, it would be easiest to start somewhat from scratch
--- and use pandas to its full extent.

Overview of the Pipeline

-- Stage 0: Locating the files and formatting
---- You will provide the path to your catalog files, so they can be located in arbitrary folders.  However, it may be easier
---- to copy and paste them into the directory that this repository is cloned into if the storage space is available.

---- It is assumed that the catalog is separated into chunks with the following file_naming convention:
------ "./catalogDirectory/file_prefix_%d.csv"   , where %d is the part (or chunk) number.
------ the catalog must contain headers for "YearID", "CrestaCode", and "GrossLoss".

-- Stage 1: catalog_constructor.py
---- The functions should not need to be touched.
---- Interact with the code below the "if__name__== "__main__":" line.
---- set the number of years in the catalog, do not use abbreviations (i.e., use 10000 for a 10K catalog, not 10K)
---- set the prefix for the file names, and include the directory name that they are contained in 
------- e.g., "./10K/10K_2018_part_"   for files contained in a folder named 10K with filenames 10K_2018_part_1, 10K_2018_part_2, ...
---- set the number of parts that exist in the file directory.

---- navigate to the folder containing catalog_constructor.py in the terminal.
---- type python catalog_constructor.py   (or python3 depending on your system's python command name, and assuming python has been added to the path)

-- Stage 2: visualize_results.py
---- Similar to the previous stage, open the file in a text editor and scroll down to the "if __name__=="__main__":" line.
---- Do not adjust the relative paths to the output files.
---- Only change the number of catalog years.  All paths are relative to what was selected in Stage 1.

-- Stage 3: locating the results
---- All outputs are stored in the "./outfiles/" folder that was created whenever catalog_constructor.py was executed.
---- "./outfiles/aal_tables/" contain csv files containing loss statistics on a county and state level.
---- These are separated into folders "./outfiles/aal_tables/counties/" and "./outfiles/aal_tables/states/" respectively.

---- "./outfiles/figures/" contains the plots that we just generated on a county and state level.
---- naming convention:
----- aal_with_error_%d_%d_years.png   : average annual loss with error bars for %d = cresta_code_for_the_county,   %d = num_catalog_years
----- Average_COV_perYear_%d_yeras.png : average coefficient of variation across all counties computed by catalog year for the %d = num_catalog_yecrs
----- frequency_of_convergence_%d_years.png:  The number of counties that reached convergence (COV < 0.05) by a particular catalog year for %d = num_catalog_yeras

---- similar files exist for the states, but no Average_COV_perYear...



-- Stage A : test_std_and_mean_calc.py
---- Contains code for testing various speedups for computing the loss statistics.  This is a major bottleneck if done improperly.
---- Currently, speedy3 is the fastest, but I cannot get the computation correct, so the values are off.  This should be an iterative std computation.
---- speedy4 is implemented in the above code, with about a 1000x time speedup over speedy.  
---- brute_force is the slowest method.
---- To test, set the number of random samples to a large number to check speed boosts.
----          set the number of random samples to a small number to compare the outputs for correctness (may need to uncomment some print statements)