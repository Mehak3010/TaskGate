# Deduplicate the rows

`/app/data/raw.csv` has the header `id,name,amount` and contains duplicate
`id` values. Duplicates must **not** simply be summed or merged: the row that
appears first in the file wins and later rows with the same `id` are dropped.

Write the deduplicated table as CSV (header included, same column order) to
`/app/out/deduplicated.csv`, sorted by `id` as an integer, ascending.
