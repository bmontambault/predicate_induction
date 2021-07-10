import numpy as np
import pandas as pd
from .data_type import Tabular

class Predicate(object):

    allowed_dtypes_map = {}
    allowed_dtypes = ['nominal', 'ordinal', 'numeric']

    def __init__(self, keys, mask=None, score=None, adjacent=None, parents=None):
        self.keys = sorted(keys)
        self.mask = mask
        if score is None:
            self.score = {}
        else:
            self.score = score
        if adjacent is None:
            self.adjacent = {}
        else:
            self.adjacent = adjacent
        self.parents = parents

    def get_mask(self, data):
        """
        return a boolean mask
        """
        raise NotImplementedError

    def get_mask_cached(self, data):
        """
        return a boolean mask
        """
        if self.mask is None:
            mask = self.get_mask(data)
            self.mask = mask
        return self.mask

    def get_score(self, data, score_f):
        """
        return score based on the data and the given scoring function
        """
        mask = self.get_mask_cached(data)
        return score_f(mask)

    def get_score_cached(self, data, score_f):
        """
        return score based on the data and the given scoring function
        """
        score_f_name = score_f.__name__
        if score_f_name not in self.score:
            self.score[score_f_name] = self.get_score(data, score_f)
        return self.score[score_f_name]

    def set_adjacent_predicate(self, key, predicate):
        if key not in self.adjacent:
            self.adjacent[key] = [predicate]
        else:
            self.adjacent[key].append(predicate)

    def set_adjacent(self, key, predicate):
        self.set_adjacent_predicate(key, predicate)
        predicate.set_adjacent_predicate(key, self)

    def is_adjacent(self, key, predicate):
        if key not in self.adjacent:
            return False
        else:
            return predicate in self.adjacent[key]

    def is_contained(self, key, predicate):
        raise NotImplementedError

    def is_subsumed(self, key, predicate, data, score_f):
        contained = self.is_contained(key, predicate)
        if not contained:
            return False
        else:
            worse = self.get_score(data , score_f) <= predicate.get_score(data, score_f)
            return worse

    def is_subsumed_any(self, key, accepted, data, score_f):
        for accepted_predicate in accepted:
            subsumed = self.is_subsumed(key, accepted_predicate, data, score_f)
            if subsumed:
                return True
        return False

    def is_subsumed_any_all_keys(self, accepted, data, score_f):
        if len(accepted) == 0:
            return False
        for key in self.keys:
            subsumed = self.is_subsumed_any(key, accepted, data, score_f)
            if subsumed:
                return True
        return False

    def merge(self, predicate, data=None):
        raise NotImplementedError

    @staticmethod
    def init_data_columns(data, data_type, allowed_dtypes, allowed_dtypes_map, dtypes=None, columns=None):
        data_obj = data_type()
        data_obj.extract(data, dtypes)
        for key, val in data_obj.dtypes.items():
            if val in allowed_dtypes_map and (columns is None or key in columns):
                data_obj.convert_dtype(key, val, allowed_dtypes_map[val])
        if columns is None:
            columns = data.columns
        columns = [col for col in columns if data_obj.dtypes[col] in allowed_dtypes]
        return data_obj, columns

    @staticmethod
    def init_predicates_top_down(data):
        raise NotImplementedError

    @staticmethod
    def init_predicates_bottom_up(data):
        raise NotImplementedError

class Conjunction(Predicate):

    allowed_dtypes_map = {'numeric': 'ordinal'}
    allowed_dtypes = ['nominal', 'ordinal']
    
    def __init__(self, column_to_values, dtypes, data=None, column_to_mask=None, mask=None, score=None, adjacent=None, parents=None):
        self.columns = list(column_to_values.keys())
        super().__init__(self.columns, mask, score, adjacent, parents)
        self.column_to_values = column_to_values
        self.dtypes = dtypes
        self.column_to_mask = column_to_mask
        if self.mask is None:
            if self.column_to_mask is not None:
                self.mask = self.get_mask_from_column_to_mask(column_to_mask)
            elif data is not None:
                self.mask = self.get_mask(data, self.column_to_values)

    def get_column_to_mask(self, column_to_values, data):
        column_to_mask = pd.DataFrame()
        for column, values in column_to_values.items():
            column_to_mask[column] = data[column].isin(values)
        return column_to_mask

    def get_mask_from_column_to_mask(self, column_to_mask):
        return column_to_mask.all(axis=1)

    def get_mask(self, data, column_to_values=None):
        if self.column_to_mask is None:
            if column_to_values is None:
                raise ValueError("column_to_values must not be None if self.column_to_mask is None")
            else:
                self.column_to_mask = self.get_column_to_mask(column_to_values, data)
        return self.get_mask_from_column_to_mask(self.column_to_mask)

    def is_contained(self, column, predicate):
        if column in self.columns and column in predicate.columns:
            return set(self.column_to_values[column]).issubset(predicate.column_to_values[column])
        else:
            return False

    def merge(self, predicate):
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
                    adjacent[column] = [p for p in self.adjacent[column] if not p.is_contained_column(predicate, column) and p not in predicate.adjacent] \
                                    + [p for p in predicate.adjacent[column] if not p.is_contained_column(self, column) and p not in self.adjacent]

        mask = self.get_mask_from_column_to_mask(column_to_mask)
        merged_predicate = Conjunction(column_to_values, self.dtypes, column_to_mask=column_to_mask, mask=mask, adjacent=adjacent, parents=[self, predicate])
        return merged_predicate

    @staticmethod
    def init_data_columns(data, dtypes=None, columns=None):
        data_obj, columns = Predicate.init_data_columns(data, Tabular, Conjunction.allowed_dtypes, Conjunction.allowed_dtypes_map, dtypes, columns)
        return data_obj, columns

    @staticmethod
    def bottom_up_init(data_obj=None, data=None, dtypes=None, columns=None):
        if data_obj is None:
            data_obj, columns = Conjunction.init_data_columns(data, dtypes, columns)
        elif columns is None:
            raise ValueError("if passing data_obj must also pass columns")
        predicates = []
        for column in columns:
            column_predicates = [Conjunction(column_to_values={column: [val]}, data=data_obj.data, dtypes=data_obj.dtypes) for val in data_obj.data[column].unique()]
            if data_obj.dtypes[column] == 'ordinal':
                for i in range(len(column_predicates)):
                    if i > 0:
                        column_predicates[i].set_adjacent(column, column_predicates[i-1])
            predicates += column_predicates
        return predicates

    # @staticmethod
    # def bottom_up_init(data_obj=None, data=None, dtypes=None, columns=None):
    #     if data_obj is None:
    #         data_obj = Conjunction.init_data(data, dtypes)
    #     if data_obj.data is None:
    #         data_obj.extract()
    #     if columns is None:
    #         columns = data_obj.data.columns
    #     columns = [col for col in columns if data_obj.dtypes[col] in ['nominal', 'ordinal', 'range']]
        
    #     predicates = []
    #     for column in columns:
    #         if data_obj.dtypes[column] == 'nominal':
    #             column_predicates = [Conjunction({column: [val]}, dtypes=data_obj.dtypes) for val in data_obj.data[column].unique()]
    #         elif data_obj.dtypes[column] == 'ordinal':
    #             column_predicates = [Conjunction({column: [val, val]}, dtypes=data_obj.dtypes) for val in sorted(data_obj.data[column].unique())]
    #         elif data_obj.dtypes[column] == 'range':
    #             column_predicates = [Conjunction({column: interval.tolist()}, dtypes=data_obj.dtypes) for interval in
    #                 data_obj.data[[f'{column}_left', f'{column}_right']].sort_values(f'{column}_left').drop_duplicates().values
    #             ]
    #         if data_obj.dtypes[column] in ['ordinal', 'range']:
    #             for i in range(len(column_predicates)):
    #                 if i > 0:
    #                     column_predicates[i].set_adjacent(column, column_predicates[i-1])
    #         predicates += column_predicates
    #     return predicates

    @staticmethod
    def initialize_predicates(search_method, data_obj=None, data=None, dtypes=None, columns=None):
        if search_method == 'bottom_up':
            return Conjunction.bottom_up_init(data_obj, data, dtypes, columns)

    def __repr__(self):
        return str(self.column_to_values)