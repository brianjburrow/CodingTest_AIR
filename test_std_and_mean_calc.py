## I have a weird way of computing the mean and standard deviation b/c we are trying
## to do it recursively.  This file aims to make sure that I coded it correctly.
import numpy as np 
import pandas 
import time
def speedy(dataset):
    nData = len(dataset)
    output = np.zeros([nData, 4]) # 4 columns, AAL, STD, STD_ERR, COV respectively

    output[:, 0] = dataset.cumsum()/np.arange(1, nData+1)

    for iData in np.arange(1, nData):
        output[iData, 1] = np.std(dataset[0:(iData + 1)] , dtype  = np.float64, ddof = 1)  # ddof makes sure that we are using the sample standard deviation
    
    output[1:, 2] = output[1:,1]/np.sqrt(np.arange(2, nData+1))
    output[1:, 3] = output[1:,2]/output[1:,0]
    return output


def speedy2(dataset):
    nData = len(dataset)
    output = np.zeros([nData, 4]) # 4 columns, AAL, STD, STD_ERR, COV respectively

    output[:, 0] = dataset["X"].cumsum()/np.arange(1, nData+1)
    output[:, 1] = dataset.expanding(1).std()["X"]
    output[1:, 2] = output[1:,1]/np.sqrt(np.arange(2, nData+1))
    output[1:, 3] = output[1:,2]/output[1:,0]
    return output


def speedy4(dataset):
    nData = len(dataset)
    output = np.zeros([nData, 4]) # 4 columns, AAL, STD, STD_ERR, COV respectively

    output[:, 0] = dataset.expanding(1).mean()["X"]
    output[:, 1] = dataset.expanding(1).std()["X"]
    output[1:, 2] = output[1:,1]/np.sqrt(np.arange(2, nData+1))
    output[1:, 3] = output[1:,2]/output[1:,0]
    return output

def speedy3(dataset):
    nData = len(dataset)
    output = np.zeros([nData, 4]) # 4 columns, AAL, STD, STD_ERR, COV respectively

    div_range = np.arange(1, nData + 1)
    output[:, 0] = dataset.cumsum()/div_range
    output[:, 1] = np.sqrt((dataset).cumsum()/(div_range - 1) - (output[:,0]/(div_range - 1))**2)
    output[1:, 2] = output[1:,1]/np.sqrt(np.arange(2, nData+1))
    output[1:, 3] = output[1:,2]/output[1:,0]
    return(output)

def brute_force(dataset):
    nData = len(dataset)
    output = np.zeros([nData, 4])
    for iData in np.arange(0, nData):
        if (iData == 0):
            output[iData, 0] = np.mean(dataset[0])
        else:
            output[iData, 0] = np.mean(dataset[0:(iData + 1)] )
            output[iData, 1] = np.std(dataset[0:(iData + 1)] , dtype  = np.float64, ddof = 1)
    return(output)
test_dataset = np.random.uniform(0, 1, 10)
df_test_dataset = pandas.DataFrame(test_dataset, columns = ["X"])

# start = time.time()
# output_speedy = speedy(test_dataset)
# print("speedy", time.time() - start)

start = time.time()
output_speedy2 = speedy2(df_test_dataset)
print("speedy2", time.time() - start)
# start = time.time()
# output_brute_force = brute_force(test_dataset)
# print("brute", time.time() - start)
# start = time.time()
# output_speedy3 = speedy3(test_dataset)
# print("speedy3",time.time() - start)

start = time.time()
output_speedy4 = speedy4(df_test_dataset)
print("speedy4", time.time() - start)
start = time.time()

# print("dataset")
# print(test_dataset)
# print("speedy")
# print(output_speedy)
# print("brute")
# print(output_brute_force)
print("speedy2")
print(output_speedy2)

print("speedy4")
print(output_speedy4)
# print("speedy3")
# print(output_speedy3)