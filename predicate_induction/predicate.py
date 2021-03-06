import pandas as pd
from .data_type import Tabular

class Predicate(object):
    """Abstract class for predicate object.

    :param keys: Variables covered by this predicate, dependent on type of data (e.g. columns for Tabular)
    :type keys: list
    :param mask: Returns subset when applied to data
    :param score: Dictionary of scores, any of which can be used during predicate induction
    :type score: dict
    :param adjacent: Dictionary mapping each key to a list of any adjacent predicates along that axis
    :type adjacent: dict
    """

    allowed_dtypes_map = {}
    allowed_dtypes = ['nominal', 'ordinal', 'numeric']

    def __init__(self, keys, mask=None, score=None, adjacent=None, is_base=False, parents=None):
        self.keys = sorted(keys)
        self.mask = mask
        self.score = score
        if adjacent is None:
            self.adjacent = {}
        else:
            self.adjacent = adjacent
        self.is_base = is_base
        self.parents = parents

    def get_mask(self, data):
        """Return a mask that will return a subset when applied to the data.
        
        :param data: Data to return mask for
        :type data: pd.DataFrame
        :return: mask
        :rtype: pd.Series
        """
        
        raise NotImplementedError

    def get_mask_cached(self, data):
        """Return a mask that will return a subset when applied to the data. Get the cached results if available.
        
        :param data: Data to return mask for
        :type data: pd.DataFrame
        :return: mask
        :rtype: pd.Series
        """

        if self.mask is None:
            mask = self.get_mask(data)
            self.mask = mask
        return self.mask

    def get_score(self, data, score_f):
        """Return a score based on the data and the given scoring function.

        :param data: Data to get score for
        :type data: pd.DataFrame
        :param score_f: Function used to calculate score
        :type score_f: function
        :return: score
        :rtype: float
        """

        mask = self.get_mask_cached(data)
        return score_f(mask)

    def get_score_cached(self, data, score_f):
        """Return a score based on the data and the given scoring function. Get the cached results if available.
        
        :param data: Data to get score for
        :type data: pd.DataFrame
        :param score_f: Function used to calculate score
        :type score_f: function
        :return: score
        :rtype: float
        """

        if self.score is None:
            self.score = self.get_score(data, score_f)
        return self.score

    def set_adjacent_predicate(self, key, predicate):
        """Set given predicate to be adjacent to this predicate along the axis of the given key.

        :param key: Axis along which predicates are adjacent
        :type key: str
        :param predicate: Adjacent predicate of the same type
        :type predicate: Predicate
        """

        if key not in self.adjacent:
            self.adjacent[key] = [predicate]
        else:
            self.adjacent[key].append(predicate)

    def set_adjacent(self, key, predicate):
        """Set this predicate and the given predicate to be adjacent to each other along the axis of the given key.

        :param key: Axis along which predicates are adjacent
        :type key: str
        :param predicate: Adjacent predicate of the same type
        :type predicate: Predicate
        """
        self.set_adjacent_predicate(key, predicate)
        predicate.set_adjacent_predicate(key, self)

    def is_adjacent(self, key, predicate):
        """Check if the given predicate is adjacent to this predicate along the given axis.

        :param key: Axis along which predicates are adjacent
        :type key: str
        :param predicate: Adjacent predicate of the same type
        :type predicate: Predicate
        :return: Whether or not the given predicate is adjacent along the given axis
        :rtype: bool
        """

        if key not in self.adjacent:
            return False
        else:
            return predicate in self.adjacent[key]

    def is_adjacent_all(self, predicate, keys=None):
        """Check if the given predicate is adjacent to this predicate along all axes.

        :param predicate: Adjacent predicate of the same type
        :type predicate: Predicate
        :param keys: Check along this axis whether this predicate is contained
        :type keys: list
        :return: Whether or not the given predicate is adjacent along all axes
        :rtype: bool
        """

        if keys is None:
            keys = self.keys
        for key in keys:
            is_adjacent_key = self.is_adjacent(key, predicate)
            if not is_adjacent_key:
                return False
        return True

    def is_worse(self, predicate, data, score_f):
        """Check if this predicate has a lower score than the given predicate.

        :param predicate: Adjacent predicate of the same type
        :type predicate: Predicate
        :param data: Data to get score for
        :type data: pd.DataFrame
        :param score_f: Function used to calculate score
        :type score_f: function
        :return: score
        :return: Whether or not this predicate has a lower score than the given predicate
        :rtype: bool
        """
        
        score = self.get_score(data , score_f)
        predicate_score = predicate.get_score(data, score_f)
        worse = score <= predicate_score
        return worse

    def is_contained_key(self, key, predicate):
        """Check if this predicate is contained by the given predicate along the given axis.

        :param key: Check along this axis whether this predicate is contained
        :type key: str
        :param predicate: Predicate of the same type that will be checked to see if it contains this predicate
        :type predicate: Predicate
        :return: Whether or not this predicate is contained by the given predicate along the given axis
        :rtype: bool
        """

        raise NotImplementedError

    def is_contained(self, predicate, keys=None):
        """Check if this predicate is contained by the given predicate along any axis.

        :param predicate: Predicate of the same type that will be checked to see if it contains this predicate
        :type predicate: Predicate
        :param keys: Check along this axis whether this predicate is contained
        :type keys: list
        :return: Whether or not this predicate is contained by the given predicate along any axis
        :rtype: bool
        """
        
        if keys is None:
            keys = predicate.keys
        for key in keys:
            if key in predicate.keys:
                is_contained_key = self.is_contained_key(key, predicate)
            else:
                is_contained_key = False
            if not is_contained_key:
                return False
        return True

    def merge(self, predicate, data=None):
        """Merge this predicate with the given predicate
        
        :param predicate: Predicate of the same type to merge with
        :param data: Data to calculate score and mask for merged predicate
        :return: Merged predicate
        """

        raise NotImplementedError

    @staticmethod
    def init_data_keys(data, data_type, allowed_dtypes, allowed_dtypes_map, dtypes=None, keys=None):
        """ Initialize data across the given keys.

        :param data: Raw data to be initialized
        :param data_type: Data type class
        :type data_type: data_type.Data
        :param allowed_dtypes: List of allowed data types
        :type allowed_dtypes: list
        :param allowed_dtypes_map: Dictionary mapping data types to allowed data types
        :type allowed_dtypes_map: dict
        :param dtypes: Dictionary mapping keys to dtype
        :type dtypes: dict
        :param keys: List of keys to include
        :type keys: list
        """

        data_obj = data_type()
        data_obj.extract(data, dtypes)
        for k, val in data_obj.dtypes.items():
            if val in allowed_dtypes_map and (keys is None or k in keys):
                data_obj.convert_dtype(k, val, allowed_dtypes_map[val])
        if keys is None:
            keys = data.keys
        keys = [k for k in keys if data_obj.dtypes[k] in allowed_dtypes]
        return data_obj, keys

    @staticmethod
    def init_predicates_top_down(data):
        """Initialized predicates for top down predicate induction.
        
        :param data: Data to initialize base predicates from
        :return: List of predicates
        :rtype: list
        """
        raise NotImplementedError

    @staticmethod
    def init_predicates_bottom_up(data):
        """Initialized predicates for bottom up predicate induction.
        
        :param data: Data to initialize base predicates from
        :return: List of predicates
        :rtype: list
        """
        raise NotImplementedError

class Conjunction(Predicate):
    """This class defines subsets of data as a conjunction over tabular data.

    :param columns_to_values: Dictionary of lists mapping columns in the data set to values defining a subset of the data
    :type column_to_values: dict
    :param dtypes: Dictionary mapping columns to dtypes
    :type dtypes: dict
    :param data: Dataframe that this predicate will take a subset of 
    :type data: pd.DataFrame
    :param column_to_mask: Dictionary mapping columns to a mask for each column, will be recalculated if None
    :type column_to_mask: dict
    :param mask: Returns subset when applied to data
    :type mask: np.array, pd.Series
    :param score: Dictionary of scores, any of which can be used during predicate induction
    :type score: dict
    :param adjacent: Dictionary mapping each key to a list of any adjacent predicates along that axis
    :type adjacent: dict
    """

    allowed_dtypes_map = {'numeric': 'ordinal'}
    allowed_dtypes = ['nominal', 'ordinal']
    
    def __init__(self, column_to_values, dtypes, data=None, column_to_mask=None, mask=None, score=None, adjacent=None, is_base=False, parents=None):
        self.columns = list(column_to_values.keys())
        super().__init__(self.columns, mask, score, adjacent, is_base, parents)
        self.column_to_values = column_to_values
        self.dtypes = dtypes
        self.column_to_mask = column_to_mask
        if self.mask is None:
            if self.column_to_mask is not None:
                self.mask = self.get_mask_from_column_to_mask(column_to_mask)
            elif data is not None:
                self.mask = self.get_mask(data, self.column_to_values)

    def get_column_to_mask(self, column_to_values, data):
        """Return a mask for each column given a dictionary of columns to values.
        
        :param columns_to_values: Dictionary of lists mapping columns in the data set to values defining a subset of the data
        :type column_to_values: dict
        :param columns_to_values: Dictionary of lists mapping columns in the data set to values defining a subset of the data
        :type column_to_values: dict
        """
        
        column_to_mask = pd.DataFrame()
        for column, values in column_to_values.items():
            column_to_mask[column] = data[column].isin(values)
        return column_to_mask

    def get_mask_from_column_to_mask(self, column_to_mask):
        """Return a mask from a dictionary mapping columns to masks.
        
        :param column_to_mask: Dictionary mapping columns to a mask for each column
        :type column_to_mask: dict
        """

        return column_to_mask.all(axis=1)

    def get_mask(self, data, column_to_values=None):
        """Return a mask from a dictionary mapping columns to values.
        
        :param columns_to_values: Dictionary of lists mapping columns in the data set to values defining a subset of the data
        :type column_to_values: dict
        """

        if self.column_to_mask is None:
            if column_to_values is None:
                raise ValueError("column_to_values must not be None if self.column_to_mask is None")
            else:
                self.column_to_mask = self.get_column_to_mask(column_to_values, data)
        return self.get_mask_from_column_to_mask(self.column_to_mask)

    def is_contained_key(self, column, predicate):
        """Check if this predicate is contained by the given predicate along the given column.

        :param column: Check along this axis whether this predicate is contained
        :type column: str
        :param predicate: Predicate of the same type that will be checked to see if it contains this predicate
        :return: Whether or not this predicate is contained by the given predicate along the given column
        :rtype: bool
        """

        if column in self.columns and column in predicate.columns:
            return set(self.column_to_values[column]).issubset(predicate.column_to_values[column])
        else:
            return False

    def merge(self, predicate):
        """Merge this predicate with the given predicate
        
        :param predicate: Predicate of the same type to merge with
        :param data: Data to calculate score and mask for merged predicate
        :return: Merged predicate
        :rtype: Conjunction
        """

        column_to_values = self.column_to_values.copy()
        column_to_mask = self.column_to_mask.copy()
        adjacent = self.adjacent.copy()
        for column, values in predicate.column_to_values.items():
            if column not in column_to_values:
                column_to_values[column] = values
                column_to_mask[column] = predicate.column_to_mask[column]
                if column in adjacent:
                    adjacent[column] = [p for p in predicate.adjacent[column] if p != self]
            else:
                column_to_values[column] = list(set(values + column_to_values[column]))
                column_to_mask[column] = column_to_mask[column] | predicate.column_to_mask[column]
                if column in adjacent:
                    adjacent[column] = [p for p in self.adjacent[column] if not p.is_contained_key(column, predicate) and p not in predicate.adjacent[column]] \
                                    + [p for p in predicate.adjacent[column] if not p.is_contained_key(column, self) and p not in self.adjacent[column]]

        mask = self.get_mask_from_column_to_mask(column_to_mask)
        merged_predicate = Conjunction(column_to_values, self.dtypes, column_to_mask=column_to_mask, mask=mask, adjacent=adjacent, parents=[self, predicate])
        return merged_predicate

    @staticmethod
    def init_data_keys(data, dtypes=None, columns=None):
        """ Initialize data for the given columns.

        :param dtypes: Dictionary mapping keys to dtype
        :type dtypes: dict
        :param columns: List of columns to include
        :type columns: list
        """

        data_obj, columns = Predicate.init_data_keys(data, Tabular, Conjunction.allowed_dtypes, Conjunction.allowed_dtypes_map, dtypes, columns)
        return data_obj, columns

    @staticmethod
    def bottom_up_init(data_obj=None, data=None, dtypes=None, columns=None):
        """Initialized predicates for bottom up predicate induction.
        
        :param data_obj: Data object used to initialize predicates, will be generated from data and dtypes if None
        :type data_obj: Data
        :param data: Data to initialize base predicates from if no data_obj is provided
        :type data: pd.DataFrame
        :param dtypes: Dictionary mapping columns to dtype if no data_obj is provided
        :type dtypes: dict
        :param columns: List of columns to include
        :type columns: list
        :return: List of predicates
        :rtype: list
        """

        if data_obj is None:
            data_obj, columns = Conjunction.init_data_keys(data, dtypes, columns)
        elif columns is None:
            raise ValueError("if passing data_obj must also pass columns")
        if data_obj.original_data is None:
            data_obj.convert_all(Conjunction.allowed_dtypes_map, Conjunction.allowed_dtypes, columns)

        predicates = []
        for column in columns:
            column_predicates = [Conjunction(column_to_values={column: [val]}, data=data_obj.data, dtypes=data_obj.dtypes, is_base=True) for val in sorted(data_obj.data[column].unique())]
            if data_obj.dtypes[column] == 'ordinal':
                for i in range(len(column_predicates)):
                    if i > 0:
                        column_predicates[i].set_adjacent(column, column_predicates[i-1])
            predicates += column_predicates
        return predicates

    def __repr__(self):
        return str(self.column_to_values)

    def __eq__(self, other):
        if isinstance(other, Conjunction):
            return self.column_to_values == other.column_to_values
        elif isinstance(other, dict):
            return self.column_to_values == other
        return False