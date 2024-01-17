In the context of many optimizers just sequentially reporting the individual evaluations
when multiple evaluations are reported at once and thus not leveraging any batch
reporting benefits, facing the concern that representing that common behaviour in the
optimizer base class requires the definition of an abstract report single and an
abstract report multi method for which the report single does not need to be implemented
if the report multi is, we decided to refactor the arising redundancy into a function
`call_functions_with_evaluations_and_collect_errors`, accepting that this increases the
cognitive load when reading the code.
