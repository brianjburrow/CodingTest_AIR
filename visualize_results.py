import pandas
import numpy as np
import glob 
import os
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.ticker as mtick

def get_filenames_by_county(base_path, n_years):
	full_path   = os.path.join(base_path, "*_%d.csv"%n_years)
	valid_files = glob.glob(full_path)
	county_cresta_codes = []
	start_remove = len(base_path) 
	for file in valid_files:
		county_cresta_codes.append(file[start_remove:start_remove+5])
	return(valid_files, county_cresta_codes)

def get_filenames_by_state(base_path, n_years):
	full_path   = os.path.join(base_path, "by_state_cresta_prefix_*_%d.csv"%n_years)
	valid_files = glob.glob(full_path)
	prefix_list = []
	start_remove = len(base_path) + len("by_state_cresta_prefix_")
	for iFile, file in enumerate(valid_files):
		prefix_list.append(valid_files[iFile][start_remove:start_remove+2])

	return valid_files, prefix_list

def load_files(file_paths, n_catalog_years):
	nCounties = len(file_paths)

	a = []*nCounties   # create a list with entries for each element in county_filenames.

	AAL_matrix     = np.zeros([n_catalog_years, nCounties])
	STD_matrix     = np.zeros([n_catalog_years, nCounties])
	STD_ERR_matrix = np.zeros([n_catalog_years, nCounties])
	COV_matrix     = np.zeros([n_catalog_years, nCounties])

	for iFile, file in enumerate(file_paths):

		temp_file                     = pandas.read_csv(file, usecols = ["AAL", "STD", "STD_ERR", "COV"], low_memory = False)

		AAL_matrix[:,iFile]           = temp_file["AAL"]
		STD_matrix[:, iFile]          = temp_file["STD"]
		STD_ERR_matrix[:, iFile]      = temp_file["STD_ERR"]
		COV_matrix[:, iFile]          = temp_file["COV"]

	COV_matrix = remove_nans(COV_matrix)
	total_output = {
	"AAL"     : AAL_matrix,
	"STD"     : STD_matrix,
	"STD_ERR" : STD_ERR_matrix,
	"COV"     : COV_matrix
	}
	return total_output

def remove_nans(numpy_array_2d):
	nrow, ncol = numpy_array_2d.shape
	for irow in np.arange(0, nrow):
		loc_of_nan                       = np.isnan(numpy_array_2d[irow,:])

		numpy_array_2d[irow, loc_of_nan] = 1
	return(numpy_array_2d)

def plot_frequency_of_convergence_by_year(cov_matrix, location_type = "states"):
	nrow, ncol = cov_matrix.shape
	frequency_of_converged = np.zeros(nrow - 1)
	years                  = np.arange(2, nrow+1)

	for irow in np.arange(1, nrow - 1):
		frequency_of_converged[irow] = sum(cov_matrix[irow, :] < 0.05)/ncol

	plt.plot(years[2:], frequency_of_converged[2:]*100, linewidth = 2.0)
	plt.ylabel("Percentage of %s converged"%location_type)
	plt.xlabel("Year")
	plt.savefig("./outfiles/figures/%s/frequency_of_convergence_%d_years.png"%(location_type, nrow) )
	plt.close()
	pass

def plot_avg_cov_by_year(cov_matrix, location_type = "states"):
	cov_matrix = cov_matrix*100 # convert to percentages
	n_years = cov_matrix.shape[0]

	avg_cov_by_year        = cov_matrix.mean(axis = 1)
	std_cov_by_year        = cov_matrix.std(axis = 1)

	fig = plt.figure()
	ax  = fig.add_subplot(111)

	ax.plot(np.arange(1, n_years + 1), avg_cov_by_year, color = 'black')
	ax.plot(np.arange(1, n_years + 1), avg_cov_by_year + 2*std_cov_by_year, color = 'red')
	ax.plot(np.arange(1, n_years + 1), avg_cov_by_year - 2*std_cov_by_year, color = 'red')
	ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
	ax.set_ylabel("Average COV Across Counties")
	ax.set_xlabel("Year")
	plt.gca().set_ylim(bottom=0)
	plt.tight_layout()

	plt.legend()
	plt.xlabel("Year")
	plt.savefig("./outfiles/figures/%s/Average_COV_perYear_%d_years.png"%(location_type, n_years) )
	plt.close()
	pass

def plot_avg_annual_loss_with_std_err(full_output_dictionary, prefix_list, location_type = "states"):
	#Dictionary should at least have entries "AAL", "STD_ERR"

	n_years, n_locations = full_output_dictionary["AAL"].shape

	for iLocation, prefix in enumerate(prefix_list):
		fig = plt.figure()
		ax  = fig.add_subplot(111)

		ax.plot(np.arange(1, n_years + 1), full_output_dictionary["AAL"][:,iLocation], color = 'black')
		ax.plot(np.arange(1, n_years + 1), full_output_dictionary["AAL"][:,iLocation] + 2*full_output_dictionary["STD_ERR"][:,iLocation], color = 'red')
		ax.plot(np.arange(1, n_years + 1), full_output_dictionary["AAL"][:,iLocation] - 2*full_output_dictionary["STD_ERR"][:,iLocation], color = 'red')
		ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
		ax.set_ylabel("Average Annual Loss")
		ax.set_xlabel("Year")
		plt.gca().set_ylim(bottom=0)
		plt.tight_layout()
		plt.savefig("./outfiles/figures/%s/aal_with_error_%s_%d_years.png"%(location_type, prefix, n_years))
		plt.close()
	pass


if __name__ == "__main__": 

	if not os.path.exists("./outfiles/figures"):
            os.mkdir("./outfiles/figures/")
            os.mkdir("./outfiles/figures/counties")
            os.mkdir("./outfiles/figures/states")

	base_path  = "./outfiles/aal_tables/counties/"  

	num_catalog_years = 10000

	filenames, county_prefixes  = get_filenames_by_county(base_path, num_catalog_years)

	output     = load_files(filenames, num_catalog_years)

	plot_avg_annual_loss_with_std_err(output, county_prefixes, location_type = "counties")
	plot_frequency_of_convergence_by_year(output["COV"], location_type = "counties")
	plot_avg_cov_by_year(output["COV"], location_type = "counties")
	#error()
	base_path  = "./outfiles/aal_tables/states/" 

	state_filenames, state_prefixes = get_filenames_by_state(base_path, num_catalog_years)

	output     = load_files(state_filenames, num_catalog_years)

	plot_avg_annual_loss_with_std_err(output, state_prefixes, location_type = "states")
	plot_frequency_of_convergence_by_year(output["COV"], location_type = "states")