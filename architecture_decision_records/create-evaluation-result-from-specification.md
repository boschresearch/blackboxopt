In the context of initializing an evaluation result from a specification, facing the
concern that having a constructor with a specification argument while the specification
attributes end up as top-level attributes and not summarized under a specification
attribute we decided for unpacking the evaluation specification like a dictionary into
the result constructor to prevent the said cognitive dissonance, accepting that the
unpacking operator can feel unintuitive and that users might tend to matching the
attributes explicitly to the init arguments.
