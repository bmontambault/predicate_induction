import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.types import Float, Integer, Text
from operator import attrgetter

class Data(object):

    original_data = None

    def __init__(self, source=None):
        raise NotImplementedError

    def infer_dtypes(self, data):
        raise NotImplementedError

    def extract(self):
        "extract data into object from source"
        raise NotImplementedError

    def load(self, data):
        "load data from object into source"
        raise NotImplementedError

    def convert_data(self, key, old_dtype, new_dtype):
        raise NotImplementedError

    def convert_dtype(self, key, old_dtype, new_dtype):
        converted_data = self.convert_data(key, old_dtype, new_dtype)
        if converted_data is not None:
            if self.original_data is None:
                self.original_data = self.data.copy()
                self.original_dtypes = self.dtypes.copy()
            self.data[key] = converted_data
            self.dtypes[key] = new_dtype

    def __call__(self, predicate):
        if self.original_data is None:
            return self.data[predicate.mask]
        else:
            return self.original_data[predicate.mask]

class Tabular(Data):

    def __init__(self, source=None, num_bins=15, num_points_per_bin=None):
        if source is None:
            self.source_type = None
        elif '.csv' in source:
            self.source_type = 'csv'
        elif 'postgresql' in source:
            self.source_type = 'postgresql'
        self.num_bins = num_bins
        self.num_points_per_bin = num_points_per_bin

    def infer_column_dtype(self, data, column):
        pd_dtype = str(data[column].dtype)
        if 'float' in pd_dtype:
            return 'numeric'
        elif 'int' in pd_dtype:
            return 'ordinal'
        else:
            return 'nominal'

    def infer_dtypes(self, data):
        return {column: self.infer_column_dtype(data, column) for column in data.columns}

    def get_table_name_engine(self, source):
        table_name = source.split('/')[-1]
        connect_string = '/'.join(source.split('/')[:-1])
        engine = create_engine(connect_string)
        return table_name, engine

    def extract_csv(self):
        data = pd.read_csv(self.source)
        return data

    def extract_postgresql(self):
        table_name, engine = self.get_table_name_engine(self.source)
        data = pd.read_sql_table(table_name, engine)
        return data

    def extract(self, data=None, dtypes=None):
        if data is None:
            if self.source is None:
                raise ValueError("must pass dataframe or source must not be None")
            elif self.source_type == 'csv':
                data = self.extract_csv()
            elif self.source_type == 'postgresql':
                data = self.extract_postgresql()
        self.data = data
        if dtypes is None:
            self.dtypes = self.infer_dtypes(self.data)
        else:
            self.dtypes = dtypes

    def load_csv(self):
        self.data.to_csv(self.source, index=False)

    def load_postgresql(self):
        table_name, engine = self.get_table_name_engine(self.source)
        sqlalchemy_dtypes = {column: Float if dtype == 'numeric'
                                     else Integer if dtype == 'ordinal'
                                     else Text for column, dtype in self.dtypes.items()}
        self.data.to_sql(table_name, engine, if_exists='replace', index=False,
                        dtype=sqlalchemy_dtypes)

    def load(self):
        if self.dtypes is None:
            self.dtypes = self.infer_dtypes(self.data)
        else:
            self.dtypes = self.dtypes
        if self.source_type is None:
            raise ValueError("need source to load")
        elif self.source_type == 'csv':
            self.load_csv()
        elif self.source_type == 'postgresql':
            self.load_postgresql()

    def bin_numeric(self, key, num_bins=15, num_points_per_bin=None):
        if num_points_per_bin is not None:
            num_bins = int(self.data.shape[0] / num_points_per_bin)
        bins = pd.cut(self.data[key], bins=num_bins)
        categories = bins.drop_duplicates().sort_values().reset_index(drop=True)
        return bins.map(pd.Series(categories.index, index=categories.values))

    def convert_data(self, key, old_dtype, new_dtype):
        if old_dtype == 'numeric' and new_dtype == 'ordinal':
            converted_data = self.bin_numeric(key, self.num_bins, self.num_points_per_bin)
        else:
            converted_data = None
        return converted_data