import numpy as np
from torch.utils import data

class Dataset(data.Dataset):

    def __init__(self, data, args, itemnum, train):
            'Initialization'
            self.data = data
            self.args = args
            self.itemnum = itemnum
            self.train = train   #  True or False

    def __len__(self):
            'Denotes the total number of samples'            
            return len(self.data)

    def __train__(self, index):
            
            session = np.asarray(self.data[index], dtype=np.int64)  
            if len(session) > self.args.maxlen:
                session = session[-self.args.maxlen:]  
            else:
                session = np.pad(session, (self.args.maxlen-len(session), 0), 'constant', constant_values=0)
            
            curr_seq = session[:-1]  # Do not take the last  one
            curr_pos = session[1:]   # Do not take the first one

            return curr_seq, curr_pos
    
    def __test__(self, index):
            # For a test set data, we also take only (maxlen - 1) actions
            # We take maxlen - 1 of them from back to front
            # padding with 0
            session = self.data[index]

            seq = np.zeros([self.args.maxlen], dtype=np.int64)
            idx = self.args.maxlen - 1  

            for i in reversed(session[:-1]): 
                seq[idx] = i
                idx -= 1
                if idx == -1: break
        
            return seq, session[-1]-1  

    def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample

            if self.train:
                return self.__train__(index)
            else:
                return self.__test__(index)
            