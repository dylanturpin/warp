Fix chained array and struct-field indexing when an index comes from array or
reference storage. Evaluate each index once as a value, preserving its target
when later index expressions or reference callees mutate that storage.
