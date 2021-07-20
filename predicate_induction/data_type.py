import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.types import Float, Integer, Text

class Data(object):
    """This is the abstract class inherited by data type classes.

    :param source: Reference to the data source (e.g. a csv or connection string)
    :type source: str
    """

    original_data = None

    def __init__(self, source=None):
        raise NotImplementedError

    def infer_dtypes(self, data):
        """Infer data types for the given data

        :param data: Data whose data types will be infered
        :return: dictionary of data types
        :rtype: dict
        """

        raise NotImplementedError

    def extract(self):
        """Extract data into object from source

        :return: data
        """

        raise NotImplementedError

    def load(self, data):
        """Load data from object into source

        :param data: Data to be loaded
        """

        raise NotImplementedError

    def convert_data(self, key, old_dtype, new_dtype):
        """Convert data from one dtype to another.

        :param key: Key in data to be converted
        :type key: str
        :param old_dtype: Dtype to be converted from
        :type old_dtype: str
        :param new_dtype: Dtype to be converted to
        :type new_dtype: str
        :return: converted data
        """

        raise NotImplementedError

    def convert_dtype(self, key, old_dtype, new_dtype):
        """Set data and dtypes to reflect dtype conversions

        :param key: Key in data to be converted
        :type key: str
        :param old_dtype: Dtype to be converted from
        :type old_dtype: str
        :param new_dtype: Dtype to be converted to
        :type new_dtype: str
        """

        converted_data = self.convert_data(key, old_dtype, new_dtype)
        if converted_data is not None:
            if self.original_data is None:
                self.original_data = self.data.copy()
                self.original_dtypes = self.dtypes.copy()
            self.data[key] = converted_data
            self.dtypes[key] = new_dtype

    def convert_all(self, allowed_dtypes_map, allowed_dtypes, keys):
        for key in keys:
            if self.dtypes[key] not in allowed_dtypes:
                self.convert_dtype(key, self.dtypes[key], allowed_dtypes_map[self.dtypes[key]])

    def __call__(self, predicate):
        """Return a subset of the data given a predicate.

        :param predicate: Predicate instance
        :type predicate: Predicate
        :return subset of the data
        """

        raise NotImplementedError

class Tabular(Data):
    """This is the class used for handling tabular data. Data from either a csv or data base connection
    will be converted to a pandas dataframe.

    :param source: Reference to the data source (e.g. a csv or connection string)
    :type source: str
    :param num_bins: Number of bins to use when converting from numeric to ordinal
    :type num_bins: int
    :param num_points_per_bin: Number of points per bin to use when converting from numeric to ordinal
    :type num_points_per_bin: int
    """

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
        """Infer data types for the given data

        :param data: Data whose data types will be infered
        :type data: pd.DataFrame
        :param column: Column whose data type will be infered
        :type column: str
        :return: dtype
        :rtype: str
        """

        pd_dtype = str(data[column].dtype)
        if data[column].isin([0,1]).all():
            return 'binary'
        if 'float' in pd_dtype:
            return 'numeric'
        elif 'int' in pd_dtype:
            return 'ordinal'
        else:
            return 'nominal'

    def infer_dtypes(self, data):
        """Infer data types for the given data

        :param data: Data whose data types will be infered
        :type data: pd.DataFrame
        :return: dictionary mapping column to data type
        :rtype: dict
        """

        return {column: self.infer_column_dtype(data, column) for column in data.columns}

    def get_table_name_engine(self, source):
        """Extract table name and engine from source

        :param source: 
        :type source: str
        :return: table_name, engine
        :rtype: str, sqlalchemy.Engine
        """

        table_name = source.split('/')[-1]
        connect_string = '/'.join(source.split('/')[:-1])
        engine = create_engine(connect_string)
        return table_name, engine

    def extract_csv(self):
        """Read csv

        :return: data
        :rtype: pd.DataFrame
        """

        data = pd.read_csv(self.source)
        return data

    def extract_postgresql(self):
        """Read from postgresql table

        :return: data
        :rtype: pd.DataFrame
        """

        table_name, engine = self.get_table_name_engine(self.source)
        data = pd.read_sql_table(table_name, engine)
        return data

    def extract(self, data=None, dtypes=None):
        """Set data from either csv or postgresql table, or from passed dataframe

        :param data: Dataframe to set as data, reads from source if this is None
        :type data: pd.DataFrame
        :param dtypes: Dictionary mapping columns to dtypes if data is not None, can be inferred if data is not None
        :type dtypes: dict 
        """

        if data is None:
            if self.source is None:
                raise ValueError("must pass dataframe or source must not be None")
            elif self.source_type == 'csv':
                data = self.extract_csv()
            elif self.source_type == 'postgresql':
                data = self.extract_postgresql()
        self.data = data
        self.keys = self.data.columns
        if dtypes is None:
            self.dtypes = self.infer_dtypes(self.data)
        else:
            self.dtypes = dtypes

    def load_csv(self):
        """Load data from source csv
        """

        self.data.to_csv(self.source, index=False)

    def load_postgresql(self):
        """Load data from source postgresql connection string
        """

        table_name, engine = self.get_table_name_engine(self.source)
        sqlalchemy_dtypes = {column: Float if dtype == 'numeric'
                                     else Integer if dtype == 'ordinal'
                                     else Text for column, dtype in self.dtypes.items()}
        self.data.to_sql(table_name, engine, if_exists='replace', index=False,
                        dtype=sqlalchemy_dtypes)

    def load(self):
        """Load data from source, either csv of postgresql connection string
        """

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

    def bin_numeric(self, column, num_bins=15, num_points_per_bin=None):
        """Bin numeric column

        :param column: The column to bin
        :type column: str
        :param num_bins: Number of bins to use when converting from numeric to ordinal
        :type num_bins: int
        :param num_points_per_bin: Number of points per bin to use when converting from numeric to ordinal
        :type num_points_per_bin: int
        """

        if num_points_per_bin is not None:
            num_bins = int(self.data.shape[0] / num_points_per_bin)
        bins = pd.cut(self.data[column], bins=num_bins)
        categories = bins.drop_duplicates().sort_values().reset_index(drop=True)
        return bins.map(pd.Series(categories.index, index=categories.values)).astype(int)

    def convert_data(self, column, old_dtype, new_dtype):
        """Convert data from one dtype to another. Currently only numeric to ordinal.

        :param column: The column in data to be converted
        :type column: str
        :param old_dtype: Dtype to be converted from
        :type old_dtype: str
        :param new_dtype: Dtype to be converted to
        :type new_dtype: str
        :return: converted data
        """

        if old_dtype == 'numeric' and new_dtype == 'ordinal':
            converted_data = self.bin_numeric(column, self.num_bins, self.num_points_per_bin)
        else:
            converted_data = None
        return converted_data

    def __call__(self, predicate):
        """Return a subset of the data given a predicate.

        :param predicate: Predicate instance
        :type predicate: Predicate
        :return: subset of the data
        """

        if self.original_data is None:
            return self.data[predicate.mask]
        else:
            return self.original_data[predicate.mask]