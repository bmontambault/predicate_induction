import numpy as np
from copy import deepcopy

class PredicateInduction(object):
    """Abstract class for predicate induction object.

    :param data: Data to search
    :param base_predicates: List of predicates to begin search
    :type base_predicates: list
    :param score_f: Function used to score predicates
    :type score_f: function
    :param frontier: List of predicates to continue search from, set to base_predicates if None
    :type frontier: list
    """

    def __init__(self, data, base_predicates, score_f, frontier=None, accepted=None, rejected=None, conditionally_accepted=None):
        self.data = data
        self.base_predicates = base_predicates
        self.score_f = score_f
        self.frontier = []
        if frontier is None:
            frontier = self.base_predicates
        for p in frontier:
            self.insert_sorted(self.frontier, p)
        if accepted is None:
            self.accepted = []
        else:
            self.accepted = accepted
        if rejected is None:
            self.rejected = []
        else:
            self.rejected = rejected
        if conditionally_accepted is None:
            self.conditionally_accepted = []
        else:
            self.conditionally_accepted = conditionally_accepted

    def get_predicate_score(self, predicate):
        """Return score for the given predicate

        :param predicate: Predicate to score
        :type predicate: Predicate
        """

        return predicate.get_score_cached(self.data, self.score_f)

    def insert_sorted(self, queue, predicate, accepted=[], verbose=False):
        """Insert the given predicate into the given queue in sorted order.
        
        :param queue: List of predicates to insert into
        :type queue: list
        :param predicate: Predicate to insert
        :type predicate: Predicate
        :param accepted: List of predicates that will be checked against to see if the predicate being inserted is subsumed
        :type accepted: list
        :param verbose: Option to print messages
        :type verbose: bool
        :return: Index where predicate was insorted
        :rtype: int
        """

        if len(queue) == 0:
            queue.append(predicate)
            return 1
        score = self.get_predicate_score(predicate)
        for i in range(len(queue)):
            i_score = self.get_predicate_score(queue[i])
            if score > i_score and not predicate.is_subsumed_any_all_keys(accepted, self.data, self.score_f):
                queue.insert(i, predicate)
                return i
        queue.append(predicate)
        return len(queue)

    def prune_subsumed(self, predicates, subsuming_predicates):
        """Prune predicates from a given list that are subsumed by predicates in the frontier.

        :param predicates: List of predicates to prune
        :type predicates: list
        :return: Pruned list of predicates
        :rtype: list
        """

        predicates = deepcopy(predicates)
        not_subsumed = []
        while len(predicates) > 0:
            predicate = predicates.pop(0)
            is_subsumed = predicate.is_subsumed_any_all_keys(subsuming_predicates, self.data, self.score_f)
            if not is_subsumed:
                not_subsumed.append(predicate)
        return not_subsumed
        
    def update_predicate_frontier(self, frontier, predicate, children):
        """Update the current frontier given a predicate and its children. If the predicate has not children and has a score greater
        than the set threshold it will be added to the list of accepted predicates. Otherwise its children will be added to the list
        of frontier predicates.

        :param frontier: Current list of frontier predicates
        :type frontier: list
        :param predicate: Original predicate
        :type predicate: Predicate
        :param children: Children of original predicate
        :type children: list
        """

        if len(children) > 0:
            for predicate in children:
                self.insert_sorted(frontier, predicate)
        elif self.get_predicate_score(predicate) > 0:
            self.insert_sorted(self.accepted, predicate, verbose=True)
        else:
            self.insert_sorted(self.rejected, predicate, verbose=True)

    def get_predicate_indices(self, predicates):
        """Get indices of given list of predicates in the frontier.

        :param predicates: Frontier predicates to get indices of
        :type predicates: list
        """

        if predicates is None:
            indices = None
        else:
            indices = [self.search.frontier.index(predicate) for predicate in predicates]
        return indices

    def update_frontier_function(self, update_f, predicates=None, predicate_indices=None, verbose=False):
        """Update the current frontier given an update function.

        :param update_f: The function used to update the current frontier
        :type update_f: function
        :param predicate_indices: Indices of frontier predicates that will be updated, all predicates will be updated if None
        :type predicate_indices: list
        :param verbose: Option to print messages
        :type verbose: bool
        """

        new_frontier = []
        if predicates is not None:
            predicate_indices = self.get_predicate_indices(predicates)
        if predicate_indices is None:
            frontier = self.frontier[:]
        else:
            frontier = self.frontier[predicate_indices]

        while len(frontier) > 0:
            predicate = frontier.pop(0)
            accepted = frontier+new_frontier+self.accepted
            expanded_predicates = update_f(predicate, accepted, verbose)
            self.update_predicate_frontier(new_frontier, predicate, expanded_predicates)
        
        if predicate_indices is None:
            self.frontier = new_frontier
        else:
            self.frontier = [self.frontier[i] for i in range(len(self.frontier)) if i not in predicate_indices] + new_frontier

        self.accepted = self.prune_subsumed(self.accepted, self.frontier)
        self.rejected = self.prune_subsumed(self.rejected, self.frontier)

    def greedy_merge_predicate(self, keys, predicate, predicates, verbose=False):
        """Merge a given predicate with a list of other predicates.

        :param predicate: Predicate to merge with other predicates
        :type predicate: Predicate
        :param predicates: List of predicates to merge
        :type predicates: list
        :param verbose: Option to print messages
        :type verbose: bool
        """
        
        score = self.get_predicate_score(predicate)
        candidate_predicates = deepcopy(predicates)
        for i in range(len(predicates)):
            p = candidate_predicates[i]
            # print(predicate, p, predicate.is_adjacent_all(p, keys))
            if predicate.is_adjacent_all(p, keys):
            # if predicate.keys == p.keys:
                merged_predicate = predicate.merge(p)
                merged_score = self.get_predicate_score(merged_predicate)
                if verbose:
                    print(predicate, score, p.score, merged_score)
                if merged_score >= score:
                    del predicates[i]
                    return self.greedy_merge_predicate(keys, merged_predicate, predicates, verbose)
            elif p.is_subsumed_any_all_keys([predicate], self.data, self.score_f, keys=keys):
            # elif p.is_subsumed(key, [predicate], self.data, self.score_f):
                del predicates[i]
        return predicate, predicates

    def greedy_merge(self, keys, predicates, threshold=0, verbose=False):
        """A single step of merging a list of predicates.
        
        :param predicates: List of predicates to merge
        :type predicates: list
        :param verbose: Option to print messages
        :type verbose: bool
        :return: List of merged predicates
        :rtype: list
        """

        merged_predicates = []
        i = 0
        while len(predicates) > 0 and i < 10000:
            predicate = predicates.pop(0)
            if predicate.score > threshold:
                predicate, predicates = self.greedy_merge_predicate(keys, predicate, predicates, verbose)
                merged_predicates.append(predicate)
                i+=1
        return merged_predicates

    def get_key_to_predicates(self, predicates):
        key_to_predicates = {}
        for predicate in predicates:
            key = tuple(predicate.keys)
            if key in key_to_predicates:
                key_to_predicates[key].append(predicate)
            else:
                key_to_predicates[key] = [predicate]
        return key_to_predicates

    def greedy_merge_frontier(self, threshold=0, verbose=False):
        print('greedy_merge_frontier')
        merged_predicates = []
        print('frontier', self.frontier)
        print()
        predicates = self.prune_subsumed(self.frontier, self.conditionally_accepted)
        print('pruned', predicates)
        key_to_predicates = self.get_key_to_predicates(predicates)
        for keys, key_predicates in key_to_predicates.items():
            keys_merged_predicates = self.greedy_merge(keys, key_predicates, threshold, verbose)
            merged_predicates += keys_merged_predicates
        return merged_predicates

    # def greedy_merge_frontier(self, verbose=False):
    #     print("greedy_merge_frontier")
    #     merged_predicates = []
    #     frontier_predicates = deepcopy(self.frontier)
    #     key_to_predicates = self.get_key_to_predicates(frontier_predicates)
    #     for key, predicates in key_to_predicates.items():
    #         key_merged_predicates = self.greedy_merge(key, predicates, verbose)
    #         for merged_predicate in key_merged_predicates:
    #             if merged_predicate.score > 0:
    #                 merged_predicates.append(merged_predicate)
    #     return merged_predicates

    def update_conditionally_accepted(self, threshold, verbose=False):
        self.conditionally_accepted = self.greedy_merge_frontier(threshold, verbose)
        self.conditionally_accepted = self.prune_subsumed(self.conditionally_accepted, self.accepted)
        self.conditionally_accepted = sorted(self.conditionally_accepted, key=lambda x: x.score, reverse=True)

class BottomUp(PredicateInduction):
    """This class is for performing bottom up predicate induction. This method involves beginning with a large number
    of predicates, each covering a small space of the data, and merging predicates into a smaller number each covering
    a larger space.

    :param data: Data to search
    :param base_predicates: List of predicates to begin search
    :type base_predicates: list
    :param score_f: Function used to score predicates
    :type score_f: function
    :param frontier: List of predicates to continue search from, set to base_predicates if None
    :type frontier: list
    """

    def __init__(self, data, base_predicates, score_f, frontier=None, accepted=None, rejected=None):
        super().__init__(data, base_predicates, score_f, frontier, accepted, rejected)
        self.keys = list(set([p.keys[0] for p in self.base_predicates]))
        self.key_to_base_predicates = {k:
            [p for p in self.base_predicates if len(p.keys) == 1 and p.keys[0] == k]
        for k in self.keys}

    def merge_predicate_candidates_key(self, predicate, key, candidate_predicates, accepted=None, verbose=False):
        """Merge the given predicate with a set of candidate predicates along a the given key.

        :param predicate: Predicate that will be merged with candidate predicates
        :type predicate: Predicate
        :param key: Axis along which predicates will be merged
        :type key: str
        :param candidate_predicates: List of predicates to merge
        :type candidate_predicates: list
        :param accepted: List of predicates that will be checked against to see if the predicate being inserted is subsumed
        :type accepted: list
        :param verbose: Option to print messages
        :type verbose: bool
        :return: New predicates
        :rtype: list
        """
        
        if candidate_predicates is None:
            return []
        if accepted is None:
            accepted = []
        new_accepted = []
        score = self.get_predicate_score(predicate)
        for candidate_predicate in candidate_predicates:
            candidate_score = self.get_predicate_score(candidate_predicate)
            merged_predicate = predicate.merge(candidate_predicate)
            merged_score = self.get_predicate_score(merged_predicate)
            if verbose:
                print(predicate, candidate_predicate, merged_predicate, score, candidate_score, merged_score)
            if merged_score > score and merged_score > candidate_score and not merged_predicate.is_subsumed_any(key, accepted, self.data, self.score_f):
                new_accepted.append(merged_predicate)
        return new_accepted

    def refine_predicate_key(self, predicate, key, accepted=None, verbose=False):
        """Refine the given predicate by merging with base predicates with the given key.

        :param predicate: Predicate that will be merged with base predicates with the given key
        :type predicate: Predicate
        :param key: Axis along which predicates will be merged
        :type key: str
        :param accepted: List of predicates that will be checked against to see if the predicate being inserted is subsumed
        :type accepted: list
        :param verbose: Option to print messages
        :type verbose: bool
        :return: New predicates
        :rtype: list
        """

        return self.merge_predicate_candidates_key(predicate, key, self.key_to_base_predicates[key], accepted, verbose)

    def expand_predicate_key(self, predicate, key, accepted=None, verbose=False):
        """Expand the given predicate by merging with adjacent predicates along the given key.

        :param predicate: Predicate that will be merged with adjacent predicates
        :type predicate: Predicate
        :param key: Axis along which predicates will be merged
        :type key: str
        :param accepted: List of predicates that will be checked against to see if the predicate being inserted is subsumed
        :type accepted: list
        :param verbose: Option to print messages
        :type verbose: bool
        :return: New predicates
        :rtype: list
        """

        return self.merge_predicate_candidates_key(predicate, key, predicate.adjacent.get(key), accepted, verbose)

    def apply_all_keys(self, predicate, keys, f, accepted=None, verbose=False):
        """Apply a function to all keys included in a given predicate.
        
        :param predicate: Predicate that the function will be applied to
        :type predicate: Predicate
        :param keys: List of keys
        :type keys: list
        :param f: Function to apply to the given predicate
        :type f: function
        :param accepted: List of predicates that will be checked against to see if the predicate being inserted is subsumed
        :type accepted: list
        :param verbose: Option to print messages
        :type verbose: bool
        :return: New predicates
        :rtype: list
        """

        return [a for b in [f(predicate, key, accepted, verbose) for key in keys] for a in b]

    def refine_predicate(self, predicate, accepted=None, verbose=False):
        """Refine the given predicate by adding additional keys that the predicate does not already include.

        :param predicate: Predicate that will be refined by adding additional keys
        :type predicate: Predicate
        :param accepted: List of predicates that will be checked against to see if the predicate being inserted is subsumed
        :type accepted: list
        :param verbose: Option to print messages
        :type verbose: bool
        :return: New predicates
        :rtype: list
        """

        return self.apply_all_keys(predicate, [key for key in self.keys if key not in predicate.keys], self.refine_predicate_key, accepted, verbose)

    def expand_predicate(self, predicate, accepted=None, verbose=False):
        """Expand the given predicate by merging with adjacent predicates along all keys.

        :param predicate: Predicate that will be merged with adjacent predicates
        :type predicate: Predicate
        :param accepted: List of predicates that will be checked against to see if the predicate being inserted is subsumed
        :type accepted: list
        :param verbose: Option to print messages
        :type verbose: bool
        :return: New predicates
        :rtype: list
        """

        return self.apply_all_keys(predicate, predicate.keys, self.expand_predicate_key, accepted, verbose)
    
    def refine_expand_predicate(self, predicate, accepted=None, verbose=False):
        """Refine and expand a given predicate.

        :param predicate: Predicate that will be refined and expanded
        :type predicate: Predicate
        :param accepted: List of predicates that will be checked against to see if the predicate being inserted is subsumed
        :type accepted: list
        :param verbose: Option to print messages
        :type verbose: bool
        :return: New predicates
        :rtype: list
        """

        return self.refine_predicate(predicate, accepted, verbose) + self.expand_predicate(predicate, accepted, verbose)

    def update_frontier_expand(self, predicates=None, predicate_indices=None, verbose=False):
        """Update the current frontier by expanding predicates.

        :param predicate_indices: Indices of frontier predicates that will be updated, all predicates will be updated if None
        :type predicate_indices: list
        :param verbose: Option to print messages
        :type verbose: bool
        """

        self.update_frontier_function(self.expand_predicate, predicates, predicate_indices, verbose)

    def update_frontier_refine(self, predicates=None, predicate_indices=None, verbose=False):
        """Update the current frontier by refining predicates.

        :param predicate_indices: Indices of frontier predicates that will be updated, all predicates will be updated if None
        :type predicate_indices: list
        :param verbose: Option to print messages
        :type verbose: bool
        """

        self.update_frontier_function(self.refine_predicate, predicates, predicate_indices, verbose)

    def update_frontier_expand_refine(self, predicates=None, predicate_indices=None, verbose=False):
        """Update the current frontier for one step.

        :param predicate_indices: Indices of frontier predicates that will be updated, all predicates will be updated if None
        :type predicate_indices: list
        :param verbose: Option to print messages
        :type verbose: bool
        """

        self.update_frontier_function(self.refine_expand_predicate, predicates, predicate_indices, verbose)