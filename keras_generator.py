import os
class Generator:
    def __init__(self,data_directory):
        self.DataDirectory=data_directory
        self.DataFiles=os.listdir(self.DataDirectory)
        TotalSize=len(self.DataFiles)*file_rows
        self.Indexes=list(range(TotalSize))
        self.index=0
    def load_training_data(self):
        for i in self.DataFiles:
            
    def preprocess():
def load_data():