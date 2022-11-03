**In the context of** ensuring deterministic behavior of optimizers, **facing the need**
to sort batches of evaluations before processing them, **we decided** to use the md5
hash of the pickled representation of relevant evaluation attributes as key for sorting
**and against** simpler approaches **to achieve** a good sorting performance,
**accepting** a slightly more complex implementation involving object serialization in
between.

_Additional background:_

The function in question, `blackboxopt.utils.sort_evaluations()`, is required to sort a
`List[Evaluation]` by only considering optimization relevant attributes and disregarding
all others that might be subject to change/randomness (e.g. timestamps).

Additionally, sorting the dictionary keys (denoted `(sorted)` in the comparison below)
is necessary for being robust against changes in the order of the reported parameters.

Sorting performance depends a lot on the amount of parameters and the number of
elements. In a quick test different methods were used to sort a list of 1000 evaluations
with 20 parameters. The time reported is total seconds for sorting the same list 1000
times (or the mean milliseconds per sort):

```
json (sorted)
32.07862658100203

md5 of json (sorted)
29.56752566599971

md5 of pickle
2.9176704900019104

md5 of pickle of (sorted) <-- pull requested implementation
8.36911787000281

md5 of str
21.312109268001223

str
19.841941792001307
```
