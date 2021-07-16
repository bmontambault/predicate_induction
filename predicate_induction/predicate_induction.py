import numpy as np
from copy import deepcopy

class PredicateInduction(object):
    """Abstract class for predicate induction object.

    :param data: Data to search
    :param base_predicates: List of predicates to begin search
    :type base_predicates: list
    :param score_f: Function used to score predicates
    :type score_f: function
    :param frontier_predicates: List of predicates to continue search from, set to base_predicates if None
    :type frontier_predicates: list
    """

    def __init__(self, data, base_predicates, score_f, frontier_predicates=None):
        self.data = data
        self.base_predicates = base_predicates
        self.score_f = score_f
        self.frontier_predicates = []
        if frontier_predicates is None:
            frontier_predicates = self.base_predicates
        for p in frontier_predicates:
            self.insert_sorted(self.frontier_predicates, p)
        self.accepted = []

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
        
    def update_predicate_frontier(self, frontier_predicates, predicate, children):
        """Update the current frontier given a predicate and its children. If the predicate has not children and has a score greater
        than the set threshold it will be added to the list of accepted predicates. Otherwise its children will be added to the list
        of frontier predicates.

        :param frontier_predicates: Current list of frontier predicates
        :type frontier_predicates: list
        :param predicate: Original predicate
        :type predicate: Predicate
        :param children: Children of original predicate
        :type children: list
        """

        if len(children) > 0:
            for predicate in children:
                self.insert_sorted(frontier_predicates, predicate)
        elif self.get_predicate_score(predicate) > 0:
            self.insert_sorted(self.accepted, predicate, verbose=True)

    def expand_frontier(self, expand_f, verbose=False):
        """Expand the current frontier given an expansion function.

        :param expand_f: The function used to expand the current frontier
        :type expand_f: function
        """

        new_frontier_predicates = []
        while len(self.frontier_predicates) > 0:
            predicate = self.frontier_predicates.pop(0)
            accepted = self.frontier_predicates+new_frontier_predicates+self.accepted
            expanded_predicates = expand_f(predicate, accepted, verbose)
            self.update_predicate_frontier(new_frontier_predicates, predicate, expanded_predicates)
        self.frontier_predicates = new_frontier_predicates

    def greedy_merge_predicate(self, predicate, predicates, verbose=False):
        """Merge a given predicate with a list of other predicates.

        :param predicate: Predicate to merge with other predicates
        :type predicate: Predicate
        :param predicates: List of predicates to merge
        :type predicates list
        :param verbose: Option to print messages
        :type verbose: bool
        """

        score = self.get_predicate_score(predicate)
        candidate_predicates = deepcopy(predicates)
        for i in range(len(predicates)):
            p = candidate_predicates[i]
            if predicate.keys == p.keys:
                merged_predicate = predicate.merge(p)
                merged_score = self.get_predicate_score(merged_predicate)
                if merged_score >= score:
                    del predicates[i]
                    return self.greedy_merge_predicate(merged_predicate, predicates, verbose)
            elif p.is_subsumed_any_all_keys([predicate], self.data, self.score_f):
                del predicates[i]
        return predicate, predicates

    def greedy_merge_step(self, predicates, verbose=False):
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
        while len(predicates) > 0 and i < 100:
            p = predicates.pop(0)
            predicate, predicates = self.greedy_merge_predicate(p, predicates, verbose)
            merged_predicates.append(predicate)
            i+=1
        return merged_predicates

    def greedy_merge(self, predicates, verbose=False):
        """Merge a list of predicates.
        
        :param predicates: List of predicates to merge
        :type predicates: list
        :param verbose: Option to print messages
        :type verbose: bool
        :return: List of merged predicates
        :rtype: list
        """

        n_predicates = -np.inf
        n_merged_predicates = np.inf
        while n_merged_predicates > 0 and n_merged_predicates != n_predicates:
            n_predicates = len(predicates)
            predicates = self.greedy_merge_step(predicates, verbose)
            n_merged_predicates = len(predicates)
        return predicates

    def expand_frontier_nsteps(self, n, verbose=False):
        """Expand the current frontier for a given number of steps

        :param n: Number of steps to expand the frontier
        :type n: int
        """
        i = 0
        while len(self.frontier_predicates) > 0 and i < n:
            self.expand_frontier(verbose)

class BottomUp(PredicateInduction):
    """This class is for performing bottom up predicate induction. This method involves beginning with a large number
    of predicates, each covering a small space of the data, and merging predicates into a smaller number each covering
    a larger space.

    :param data: Data to search
    :param base_predicates: List of predicates to begin search
    :type base_predicates: list
    :param score_f: Function used to score predicates
    :type score_f: function
    :param frontier_predicates: List of predicates to continue search from, set to base_predicates if None
    :type frontier_predicates: list
    """

    def __init__(self, data, base_predicates, score_f, frontier_predicates=None):
        super().__init__(data, base_predicates, score_f, frontier_predicates)
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
            merged_predicate = predicate.merge(candidate_predicate)
            merged_score = self.get_predicate_score(merged_predicate)
            if verbose:
                print(merged_predicate, merged_score, score)
                print(merged_score > score, not merged_predicate.is_subsumed_any(key, accepted, self.data, self.score_f))
            if merged_score > score and not merged_predicate.is_subsumed_any(key, accepted, self.data, self.score_f):
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

    def merge_adjacent_predicate_key(self, predicate, key, accepted=None, verbose=False):
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

    def merge_adjacent_predicate(self, predicate, accepted=None, verbose=False):
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

        return self.apply_all_keys(predicate, predicate.keys, self.merge_adjacent_predicate_key, accepted, verbose)
    
    def refine_merge_adjacent_predicate(self, predicate, accepted=None, verbose=False):
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

        return self.refine_predicate(predicate, accepted, verbose) + self.merge_adjacent_predicate(predicate, accepted, verbose)

    def expand_frontier(self, verbose=False):
        """Expand the current frontier for one step.
        """

        super().expand_frontier(self.refine_merge_adjacent_predicate, verbose)