import numpy as np
from copy import deepcopy
from .predicate import Conjunction

class PredicateInduction(object):

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
        return predicate.get_score_cached(self.data, self.score_f)

    def insert_sorted(self, queue, predicate, accepted=[], verbose=False):
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
        
    def update_predicate_frontier(self, frontier_predicates, predicate, new_predicates):
        if len(new_predicates) > 0:
            for predicate in new_predicates:
                self.insert_sorted(frontier_predicates, predicate)
        elif self.get_predicate_score(predicate) > 0:
            self.insert_sorted(self.accepted, predicate, verbose=True)

    def expand_frontier(self, expand_f, verbose=False):
        new_frontier_predicates = []
        while len(self.frontier_predicates) > 0:
            predicate = self.frontier_predicates.pop(0)
            accepted = self.frontier_predicates+new_frontier_predicates+self.accepted
            expanded_predicates = expand_f(predicate, accepted, verbose)
            self.update_predicate_frontier(new_frontier_predicates, predicate, expanded_predicates)
        self.frontier_predicates = new_frontier_predicates

    def greedy_merge_predicate(self, predicate, predicates, verbose=False):
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
        merged_predicates = []
        i = 0
        while len(predicates) > 0 and i < 100:
            p = predicates.pop(0)
            predicate, predicates = self.greedy_merge_predicate(p, predicates, verbose)
            merged_predicates.append(predicate)
            i+=1
        return merged_predicates

    def greedy_merge(self, predicates, verbose=False):
        n_predicates = -np.inf
        n_merged_predicates = np.inf
        while n_merged_predicates > 0 and n_merged_predicates != n_predicates:
            n_predicates = len(predicates)
            predicates = self.greedy_merge_step(predicates, verbose)
            n_merged_predicates = len(predicates)
        return predicates

    def expand_frontier_nsteps(self, n, verbose=False):
        i = 0
        while len(self.frontier_predicates) > 0 and i < n:
            self.expand_frontier(verbose)

class BottomUp(PredicateInduction):

    def __init__(self, data, base_predicates, score_f, frontier_predicates=None):
        super().__init__(data, base_predicates, score_f, frontier_predicates)
        self.keys = list(set([p.keys[0] for p in self.base_predicates]))
        self.key_to_base_predicates = {k:
            [p for p in self.base_predicates if len(p.keys) == 1 and p.keys[0] == k]
        for k in self.keys}

    def merge_predicate_candidates_key(self, predicate, key, candidate_predicates, accepted=None, verbose=False):
        if verbose:
            print('merge_predicate_candidate_key')
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
        return self.merge_predicate_candidates_key(predicate, key, self.key_to_base_predicates[key], accepted, verbose)

    def merge_adjacent_predicate_key(self, predicate, key, accepted=None, verbose=False):
        return self.merge_predicate_candidates_key(predicate, key, predicate.adjacent.get(key), accepted, verbose)

    def apply_all_keys(self, predicate, keys, f, accepted=None, verbose=False):
        return [a for b in [f(predicate, key, accepted, verbose) for key in keys] for a in b]

    def refine_predicate(self, predicate, accepted=None, verbose=False):
        return self.apply_all_keys(predicate, [key for key in self.keys if key not in predicate.keys], self.refine_predicate_key, accepted, verbose)

    def merge_adjacent_predicate(self, predicate, accepted=None, verbose=False):
        return self.apply_all_keys(predicate, predicate.keys, self.merge_adjacent_predicate_key, accepted, verbose)
    
    def refine_merge_adjacent_predicate(self, predicate, accepted=None, verbose=False):
        return self.refine_predicate(predicate, accepted, verbose) + self.merge_adjacent_predicate(predicate, accepted, verbose)

    def expand_frontier(self, verbose=False):
        super().expand_frontier(self.refine_merge_adjacent_predicate, verbose)