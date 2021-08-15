# Predicate Induction

[Documentation](https://htmlpreview.github.io/?https://github.com/bmontambault/predicate_induction/blob/master/docs/build/html/predicate_induction.html)

# Bottom-up Algorithm

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