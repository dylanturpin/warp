Preserve bounded ``CountParameter`` operands in recorded CUDA launch dimensions,
int32 scalar arguments, and explicitly declared copy/fill/zero prefixes. Retain
operand identity and resource ownership in canonical records, and block replay
until external binding preparation completes. Use ``upper_bound`` for explicit
static planning; ordinary integer launches and omitted whole-array extents retain
their previous meaning.
