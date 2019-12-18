# Documentation for `tag_seq`

## What is `tag_seq`?
A `tag_seq` records the semantic meaning of the parser's every decision, including:
- its category (e.g., `SELECT_AGG`), 
- content (e.g., the specific aggregator), 
- contextual information (e.g., the corresponding column of this aggregator), and
- some meta information (e.g., the index of this decision in action space, the decision probability).

Each item in a `tag_seq` is usually called a `semantic unit`. A semantic unit can be about the parser's decisions on `SELECT_COL`, `SELECT_AGG`, etc.

## How should `tag_seq` be generated?
A `tag_seq` is usually generated while decoding a SQL query, especially when the decoding is grammar-based (so that, for instance, one can easily know the category and meta information). 
See [SQLNet's tag_seq implementation](https://github.com/sunlab-osu/MISP/blob/multichoice_q/SQLNet_model/sqlnet/model/sqlnet.py#L247).

## How will `tag_seq` be used?
A `tag_seq` will be fed to the `Error Detector (ED)` to find potential mistakes and `Question Generator (QG)` to form questions. 

Basically, the ED module reads the `semantic units` one by one and decides whether the decision is likely to be wrong by examining its meta information.

When a semantic unit is deemed wrong, it will be passed to QG. QG generates a question about this decision by associating the category and content of this decision with its contextual information - all have been recorded in the `tag_seq`!


## Semantic Unit Definitions
Our framework already defines a comprehensive list of semantic units for common use, as shown below. All _tags (e.g., SELECT_COL, OUTSIDE)_ are defined [here](MISP_SQL/utils.py#L6).

Note that one can also define their own units to meet different needs. We will give some examples.

### Basics
- Column `col`: `col = (tab_name, col_name, col_idx)` # `col_idx` is the index of this column in the column/action space. Such indices will be used to compare with the golden query (when simulating user feedback).
- Aggregator `agg`: `agg = (agg_name, agg_idx) # agg_name in {'min', 'max', 'count', ..., "none_agg" # empty agg}`
- Operator `op`: `op = (op_name, op_idx) # op_name in {'>', '<', ...}`
- Ordering `desc_asc_limit`: `desc_asc_limit = ('asc'/'desc', True/False for limit)`
- Intersect, Union, Except, None: `iuen = (iuen_name, iuen_idx) # iuen_name in {'none', 'intersect', 'union', 'except'}`

### SELECT clause
- `(SELECT_COL, col, p(col), dec_idx)` # note that `col` has to follow the aforementioned definition; `dec_idx` is the index of this decision in `dec_seq` (see [the introduction](https://github.com/sunlab-osu/MISP#2-system-architecture)).
- `(SELECT_AGG, col, agg, p(agg), dec_idx)` # note that `agg` has to follow the aforementioned definition; `dec_idx` is the index of this decision (predicting `agg`, not `col`) in `dec_seq`.

### WHERE clause
- `(WHERE_COL, col, p(col), dec_idx)`
- `(WHERE_OP, (col,), op, p(op), dec_idx)`
- `(WHERE_VAL, (col,), op, (val_idx, val_str), p(val_str), dec_idx)` # used in WikiSQL; val_idx is the list of word indices
For Spider:
- `(WHERE_ROOTTERM, (col,), op, 'root'/'terminal', p('root'/'terminal'), dec_idx)` # used in Spider
- `(ANDOR, 'and'/'or', [col1, col2, ..], p('and'/'or'), dec_idx)` # [col1, col2, ..] are columns selected in WHERE clause

### GROUP BY and HAVING clause
GROUP BY:
- `(GROUP_COL, col, p(col), dec_idx)`
- (Optional; model-specific) We also added the following definitions for SyntaxSQLNet:

`(GROUP_NHAV, "none_having", p("none_having"), dec_idx)` # SyntaxSQLNet has a particular decision on whether to add a HAVING clause; we thus define this unit so this decision can be validated by users as well.

Note that, the following units about HAVING have to be placed after GROUP BY:
- `(HAV_COL, col, p(col), dec_idx)`
- `(HAV_AGG, col, agg, p(agg), dec_idx)`
- `(HAV_OP, (col, agg), op, p(op), dec_idx)`
- `(HAV_ROOTTERM, (col, agg), op, 'root'/'terminal', p('root'/'terminal'), dec_idx)`

### ORDER BY clause
- `(ORDER_COL, col, p(col), dec_idx)`
- `(ORDER_AGG, col, agg, p(agg), dec_idx)`
- `(ORDER_DESC_ASC_LIMIT, (col, agg), desc_asc_limit, p(desc_asc_limit), dec_idx)`


### Intersect, Union, Except, None
- `('IUEN', iuen, p(iuen), dec_idx)` # iuen

### Nested queries
**Case 1:**
Our framework allows generating questions for nested queries, such as `SELECT ... WHERE col1 = ( <- this is a nested query -> )`.
However, one has to append a unit `(OUTSIDE, '##END_NESTED##', 1.0, None)` after completing a nested query:
```
<- here are units for the main sql-> .. (WHERE_OP, (col,), op, p(op), dec_idx),
(WHERE_ROOTTERM, (col,), op, 'root', p('root'), dec_idx), <- here are units for the nested sql -> (OUTSIDE, '##END_NESTED##', 1.0, None), 
<- here are units for the demaining main sql->
```

**Case 2:**
Nested queries can also happen to queries with Intersect/Union/Except.

For models like SyntaxSQLNet which first decides `iuen` then generates the main or nested query, its `tag_seq` should look like:
```
('IUEN', iuen, p(iuen), dec_idx), <- here are units for the main sql -> (OUTSIDE, '##END_NESTED##', 1.0, None),
<- followed by units for the nested sql ->
```

For models decoding a SQL query token-by-token (e.g., EditSQL), `##END_NESTED##` may not be necessary, and the `tag_seq` can look like:
```
<- here are units for the main sql -> 
('IUEN', iuen, p(iuen), dec_idx) 
<- followed by units for the nested sql ->
```






