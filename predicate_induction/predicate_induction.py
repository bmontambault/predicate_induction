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

    def update_frontier(self, parent, children, verbose=False, tracked_predicates=None):
        is_done = True
        if verbose and (tracked_predicates is None or parent in tracked_predicates):
            children_added_to_frontier = []
            children_subsumed_by = []
        all_children_subsumed = len(children) > 0
        for child in children:
            if child.score > parent.score:
                is_subsumed = False
                i = 0
                while not is_subsumed and i < len(self.accepted):
                    if child.is_contained(self.accepted[i]):
                        if self.accepted[i].score > child.score:
                            is_subsumed = True
                            subsuming_keys = self.accepted[i].keys
                    i+=1
                if verbose and (tracked_predicates is None or parent in tracked_predicates):
                    children_subsumed_by.append((child, self.accepted[i-1]))
                if verbose and (tracked_predicates is None or child in tracked_predicates):
                    if not is_subsumed:
                        print(child, 'added to frontier')
                    else:                            
                        print(child, 'subsumed, NOT added to frontier')
                if not is_subsumed:
                    if verbose and (tracked_predicates is None or parent in tracked_predicates):
                        children_added_to_frontier.append(child)
                    is_done = False
                    self.insert_sorted(self.frontier, child)
                    all_children_subsumed = False
                if is_subsumed and not any(key in subsuming_keys for key in parent.keys):
                    all_children_subsumed = False
            else:
                all_children_subsumed = False
        if verbose and (tracked_predicates is None or parent in tracked_predicates):
            print('children:', children_added_to_frontier, 'parent done:', is_done, 'children subsumed by:', children_subsumed_by, 'all children subsumed:', all_children_subsumed)
        return is_done, all_children_subsumed

    def insert_sorted(self, queue, predicate):
        if len(queue) == 0:
            queue.append(predicate)
            return 1
        score = self.get_predicate_score(predicate)
        for i in range(len(queue)):
            i_score = self.get_predicate_score(queue[i])
            if score > i_score:
                queue.insert(i, predicate)
                return i
        queue.append(predicate)
        return len(queue)

    def move_predicate(self, predicate, location, destination):
        location.remove(predicate)
        self.insert_sorted(destination, predicate)

    def update_accepted_rejected_predicate(self, predicate, is_done, all_children_subsumed, threshold=0, verbose=False, tracked_predicates=None):
        if is_done:
            if all_children_subsumed:
                if verbose and (tracked_predicates is None or predicate in tracked_predicates):
                    print(predicate, 'added to rejected')
                self.insert_sorted(self.rejected, predicate)
            else:
                if predicate.score > threshold:
                    is_subsuming = True
                    contained_predicates = [p for p in self.accepted if p.is_contained(predicate)]
                    i = 0
                    while is_subsuming and i < len(contained_predicates):
                        if predicate.score <= contained_predicates[i].score:
                            self.insert_sorted(self.rejected, predicate)
                            is_subsuming = False
                        i+=1
                    if is_subsuming and predicate.score > threshold:
                        is_subsumed = False
                        j = 0
                        while predicate.is_base and not is_subsumed and j < len(self.accepted):
                            if predicate.is_contained(self.accepted[j]) and predicate.score <= self.accepted[j].score:
                                is_subsumed = True
                            j+=1
                        if not is_subsumed:
                            self.insert_sorted(self.accepted, predicate)
                            if verbose and (tracked_predicates is None or predicate in tracked_predicates):
                                print(predicate, 'added to accepted')
                        else:
                            self.insert_sorted(self.rejected, predicate)
                            if verbose and (tracked_predicates is None or predicate in tracked_predicates):
                                print(predicate, 'subsumed; added to rejected')
                        for contained_predicate in contained_predicates:
                            if verbose and (tracked_predicates is None or contained_predicate in tracked_predicates):
                                print(contained_predicate, 'moved to rejected')
                            self.move_predicate(contained_predicate, self.accepted, self.rejected)
                else:
                    self.insert_sorted(self.rejected, predicate)

    def greedy_merge_predicate(self, keys, predicate, predicates, verbose=False, tracked_predicates=None):
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
                merged_predicate = predicate.merge(p)
                merged_score = self.get_predicate_score(merged_predicate)
                if verbose and (predicate is None or predicate in tracked_predicates):
                    print(predicate, score, p.score, merged_score)
                if merged_score >= score:
                    del predicates[i]
                    return self.greedy_merge_predicate(keys, merged_predicate, predicates, verbose, tracked_predicates)
            elif p.is_subsumed(predicate, self.data, self.score_f, keys=keys):
                del predicates[i]
        return predicate, predicates

    def greedy_merge(self, keys, predicates, threshold=0, verbose=False, tracked_predicates=None):
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
        while len(predicates) > 0 and i < 100000:
            predicate = predicates.pop(0)
            if predicate.score > threshold:
                predicate, predicates = self.greedy_merge_predicate(keys, predicate, predicates, verbose, tracked_predicates)
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

    def greedy_merge_frontier(self, threshold=0, verbose=False, tracked_predicates=None):
        merged_predicates = []
        key_to_predicates = self.get_key_to_predicates(self.frontier)
        for keys, key_predicates in key_to_predicates.items():
            pruned_key_predicates = []
            for predicate in key_predicates:
                is_subsumed = False
                i=0
                while i < len(self.conditionally_accepted) and not is_subsumed:
                    is_contained = predicate.is_contained_key(self.conditionally_accepted[i])
                    if is_contained and self.conditionally_accepted[i] > predicate.score:
                        is_subsumed = True
                    i+=1
                if not is_subsumed:
                    pruned_key_predicates.append(predicate)
            keys_merged_predicates = self.greedy_merge(keys, pruned_key_predicates, threshold, verbose, tracked_predicates)
            merged_predicates += keys_merged_predicates
        return merged_predicates

    def get_conditionally_accepted(self, conditional_threshold, verbose, tracked_predicates):
        if conditional_threshold is not None:
            conditionally_accepted = self.greedy_merge_frontier(conditional_threshold, verbose, tracked_predicates)
        else:
            conditionally_accepted = []
        return conditionally_accepted

    def merge_predicates(self, predicates_a, predicates_b):
        merged_predicates = []
        keep_a = [True for i in range(len(predicates_a))]
        keep_b = [True for j in range(len(predicates_b))]
        for i in range(len(predicates_a)):
            for j in range(len(predicates_b)):
                if predicates_a[i].is_contained(predicates_b[j]) or predicates_b[j].is_contained(predicates_a[i]):
                    if predicates_a[i].score >= predicates_b[j].score:
                        keep_b[j] = False
                    elif predicates_b[j].score >= predicates_a[i].score:
                        keep_a[i] = False
        for i in range(len(predicates_a)):
            if keep_a[i]:
                self.insert_sorted(merged_predicates, predicates_a[i])
        for j in range(len(predicates_b)):
            if keep_b[j]:
                self.insert_sorted(merged_predicates, predicates_b[j])
        return merged_predicates

    def get_first_index(self, predicates=None):
        if predicates is None:
            first_index = 0
        else:
            first_index = [self.frontier.index(predicate) for predicate in sorted(predicates, key=lambda x: x.score, reverse=True)][0]
        return first_index

    def update_accepted_rejected_function(self, update_f, predicates=None, threshold=0, verbose=False, tracked_predicates=None):
        first_index = self.get_first_index(predicates)
        predicate = self.frontier.pop(first_index)            
        children = update_f(predicate, verbose, tracked_predicates)
        is_done, all_children_subsumed = self.update_frontier(predicate, children, verbose, tracked_predicates)
        self.update_accepted_rejected_predicate(predicate, is_done, all_children_subsumed, threshold, verbose, tracked_predicates)

    def get_predicates(self, conditional_threshold, verbose=False, tracked_predicates=None):
        conditionally_accepted = self.get_conditionally_accepted(conditional_threshold, verbose, tracked_predicates)
        merged_accepted = self.merge_predicates(self.accepted, conditionally_accepted)
        return merged_accepted

    def get_max_score(self):
        return max(self.frontier[0].score, self.accepted[0].score)

    def get_predicates_maxiters(self, update_f, predicates=None, maxiters=None, threshold=0, conditional_threshold=None, verbose=False, tracked_predicates=None):
        i = 0
        while len(self.frontier) > 0 and (maxiters is None or i < maxiters):
            if conditional_threshold is not None:
                max_score = self.get_max_score()
                if max_score > conditional_threshold:
                    return self.get_predicates(conditional_threshold, verbose, tracked_predicates)
            self.update_accepted_rejected_function(update_f, predicates, threshold, verbose, tracked_predicates)
            i+=1
        return self.get_predicates(conditional_threshold, verbose, tracked_predicates)


class BottomUp(PredicateInduction):
    """This class is for performing bottom up predicate induction. This method involves beginning with a large number
    of predicates, each covering a small space of the data, and merging predicates into a smaller number each covering
    a larger space.
    -- Set threshold and conditionalThreshold
    -- Initialize empty queues F (frontier), A (accepted), and R (rejected)
    -- Add each base predicate to F
    -- While F is not empty and i < max_iters:
        -- For each predicate p in F:
            -- let DONE = True
            -- Remove p from F
            -- Generate a list of new predicates N by:
                -- expanding: merge p with adjacent predicates along each dimension
                -- refining: merge p with predicates along uncovered dimension
            -- For each new predicate n:
                -- If score(n) > score(p):
                    -- let DONE = False
                    -- let SUBSUMED = False
                    -- Check for all predicates in A that fully contain n (include the same dimensions as pand all points are included in n) and add them to the queue C
                    -- For c in C:
                        -- If score(c) > score(n) add n to R, let SUBSUMED = TRUE, and stop inner loop
                    -- If not SUBSUMED add n to F
            -- If DONE:
                -- If the score(p) > threshold:
                    -- Check for predicates in A that are fully contained by p and add them to an empty queue B
                    -- let SUBSUMING = True
                    -- For b in B:
                        -- If score(b) >= score(p) add p to R, let SUBSUMING = False, and stop inner loop
                    -- If SUBSUMING move all b in B from A to R and add p to A
                -- Else add p to R
    -- Initialize empty queue CA (conditionally accepted)
    -- If F is not empty
        -- let greedyMerge = function(p, sortedQueue){
            -- For q in sortedQueue:
                -- let m = p.merge(q)
                -- If score(m) > score(q):
                    -- remove q from sortedQueue
                    -- greedyMerge(m, sortedQueue)
                -- Else return [p, sortedQueue]
        }
        -- Initialize set of k empty queues K for k sets of predicates in F with the same dimensions
        -- For f in F:
            -- add f to K[f.dimensions] in sorted order
        -- For k in K:
            -- While k is not empty:
                -- let k0 = k[0]
                -- remove k0 from k
                -- let k0, k = greedyMerge(k0, k)
                -- If score(k0) > conditionalThreshold add k0 to CA
    -- Initialize empty queue FINAL
    -- let keepA = {True | a in A}
    -- let keepCA = {True | ca in CA}
    -- For a in A:
        -- For ca in CA:
            -- If a.contains(ca) or ca.contains(a)
                -- If score(a) >= score(ca) let keepCA[ca.index] = False
                -- If score(a) < score(ca) let keepA[a.index] = False
    -- For a in A:
        -- If keepA[a.index] add a to FINAL
    -- For ca in CA:
        -- If keepCA[ca.index] add ca to FINAL
    -- return sorted(FINAL)

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

    def merge_predicate_candidates(self, predicate, candidate_predicates, verbose=False, tracked_predicates=None):
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
        children = []
        score = self.get_predicate_score(predicate)
        for candidate_predicate in candidate_predicates:
            candidate_score = self.get_predicate_score(candidate_predicate)
            merged_predicate = predicate.merge(candidate_predicate)
            merged_score = self.get_predicate_score(merged_predicate)
            if verbose and (tracked_predicates is None or predicate in tracked_predicates or merged_predicate in tracked_predicates):
                print(predicate, candidate_predicate, merged_predicate, score, candidate_score, merged_score)
            if merged_score > score:
                children.append(merged_predicate)
        return children

    def refine_predicate_key(self, predicate, key, verbose=False, tracked_predicates=None):
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

        return self.merge_predicate_candidates(predicate, self.key_to_base_predicates[key], verbose, tracked_predicates)

    def expand_predicate_key(self, predicate, key, verbose=False, tracked_predicates=None):
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

        return self.merge_predicate_candidates(predicate, predicate.adjacent.get(key), verbose, tracked_predicates)

    def apply_all_keys(self, predicate, keys, f, verbose=False, tracked_predicates=None):
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

        return [a for b in [f(predicate, key, verbose, tracked_predicates) for key in keys] for a in b]

    def refine_predicate(self, predicate, verbose=False, tracked_predicates=None):
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

        return self.apply_all_keys(predicate, [key for key in self.keys if key not in predicate.keys], self.refine_predicate_key, verbose, tracked_predicates)

    def expand_predicate(self, predicate, verbose=False, tracked_predicates=None):
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

        return self.apply_all_keys(predicate, predicate.keys, self.expand_predicate_key, verbose, tracked_predicates)
    
    def expand_refine_predicate(self, predicate, verbose=False, tracked_predicates=None):
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

        return self.expand_predicate(predicate, verbose, tracked_predicates) + self.refine_predicate(predicate, verbose, tracked_predicates)

    def expand(self, predicates=None, maxiters=None, threshold=0, conditional_threshold=None, verbose=False, tracked_predicates=None):
        return self.get_predicates_maxiters(self.expand_predicate, predicates, maxiters, threshold, conditional_threshold, verbose, tracked_predicates)

    def refine(self, predicates=None, maxiters=None, threshold=0, conditional_threshold=None,verbose=False, tracked_predicates=None):
        return self.get_predicates_maxiters(self.refine_predicate, predicates, maxiters, threshold, conditional_threshold, verbose, tracked_predicates)

    def expand_refine(self, predicates=None, maxiters=None, threshold=0, conditional_threshold=None,verbose=False, tracked_predicates=None):
        return self.get_predicates_maxiters(self.expand_refine_predicate, predicates, maxiters, threshold, conditional_threshold, verbose, tracked_predicates)