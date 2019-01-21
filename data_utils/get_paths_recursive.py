import numpy as np
import os
def get_pkl(directory = "", max_depth=-1, loadingType='pkl'):
    """ 
    @Author: Salman Ahmed
    Recursively iterate over all nested directories and get paths of PKL files in directory and sub-directories.
    directory is PATH from where it will get all paths.
    max_depth is depth limit given as argument to how much deep we want to go.
    max_depth = -1 means till end.
    """
    lists = [] #List containing all paths 
    if max_depth==0: #Terminating condition for depth
        return []
    
    lstDir = os.listdir(directory) #Getting all files and directories in given directory
    for each in lstDir: #Iterating over each file or directory in list
        
        checkEst = each.split('.') #Splitting by (.) to know if current file in iteration is PKL file, a directory or anything else

        if len(checkEst)==1 and  os.path.isdir(os.path.join(directory,checkEst[0])): #Checking if it is directory
            lists.extend(get_pkl(os.path.join(directory,checkEst[0]), max_depth-1, loadingType)) #If directory then calling same function recursively and subtracting 1 from depth
        
        if checkEst[-1] == loadingType or checkEst[-1] == loadingType.upper() or checkEst[-1] == loadingType.lower(): #If current file is a PKL file 
            lists.append(os.path.join(directory,each)) #Appending PKL file to lists
    return lists #Returning all PKL files