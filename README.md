# deferkv

Numpy implementation of the decode-aware KV importance score from DeferKV ([arXiv:2610.06286](https://arxiv.org/abs/2610.06286)).

## Install

```bash
pip install -e ".[test]"
```

## Run

```bash
python -m pytest -q
python demo.py
```

## What does not match the paper

No LLM and no LongBench or RULER numbers. This is the scoring rule only. `s_j = sum_i eta^{W-i} a_{i,j}` with `eta = 0.7`, the first decode query at weight 1, and average pooling with kernel size 5. The historical candidate set is `{1, ..., N-W+1}` as in Appendix A, and the suffix outside that set is always kept. Head-wise budget allocation from SnapKV is not implemented; `budget` is a single top-k over the candidate set. Compression is assumed to happen after the first decode query, which is the input window, not a generation loop.
