from torch.utils.data import Dataset


class ChessDataSet(Dataset):
    def __init__(self, database, num_indexes):
        super(Dataset).__init__()

        self.database = database

        # check if num indexes is possible with size of current dataset if not throw an error
        if database.size() < num_indexes:
            raise ValueError(
                "num_indexes is larger than the size of the database\nIncrease the size of the database through the config file or decrease num_indexes"
            )
        self.num_indexes = num_indexes

    def __len__(self):
        return self.num_indexes

    def __getitem__(self, id):
        return self.database.get_item(id)

    def __getitems__(self, ids):
        # this is a modification of the __getitem__ method to retrieve multiple indexes for batch processing.
        return [self.database.get_item(item) for item in ids]
