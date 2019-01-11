class Generator:
    def __init__(self, config):
        self.config = config
        self.DataDirectory = self.config['data_directory']
        self.DataFiles = np.array(os.listdir(self.DataDirectory))
        self.Indexes_files = np.array(list(range(len(self.DataFiles))))
        self.Data = []
        self.i_e = 0
        self.i_f = 0

    def fill_buffer(self):
        files_batch_size = self.config['file_batch_size']
        files = self.DataFiles[self.Indexes_files[self.i_f:self.i_f + files_batch_size]]
        self.Data = self.load_files(files, self.config['format_'])
        self.i_f += files_batch_size

    def load_files(self, files, format_):
        data_list = []
        if format_ == 'csv':
            for i in files:
                if i[-4:] == '.csv':
                    array = pd.read_csv(os.path.join(self.DataDirectory, i)).values
                    data_list.extend(array)
        elif format_ == 'h5':
            for i in files:
                if i[-3:] == '.h5':
                    array = pd.read_hdf(os.path.join(self.DataDirectory, i)).values

                    data_list.extend(array)
        elif format_ == 'npy':
            for i in files:
                if i[-4:] == '.csv':
                    array = np.load(os.path.join(self.DataDirectory, i))
                    data_list.extend(array)
        return np.array(data_list)

    def load_data(self):
        self.examples_batch_size = self.config['ex_batch_size']
        if self.i_f >= len(self.Indexes_files):
            np.random.shuffle(self.Indexes_files)
            self.i_f = 0
        if self.i_e >= len(self.Data):
            self.fill_buffer()
            self.indexes_examples = np.arange(0, len(self.Data), dtype=np.int32)
            np.random.shuffle(self.indexes_examples)
            self.i_e = 0
        res = self.Data[self.indexes_examples[self.i_e:self.i_e + self.examples_batch_size]]
        self.i_e += self.examples_batch_size
        return res

    def __len__(self):
        return (len(self.DataFiles) * self.config['file_examples']) // self.config['ex_batch_size']

    def on_epoch_end(self):
        pass

    def __getitem__(self, idx):
        data = self.load_data()
        print(data.shape)
        data1 = data[:, np.arange(0, len(self.Data[0])) != self.config['obsnet_col']]
        obsnet = data[:, np.arange(0, len(data[0])) == self.config['obsnet_col']]
        list_obsnet = []
        for i in obsnet:
            list_obsnet.append(np.array(i[0]).flatten())
        obsnet = np.array(list_obsnet)
        return np.c_[data1, obsnet], self.i_e, self.i_f