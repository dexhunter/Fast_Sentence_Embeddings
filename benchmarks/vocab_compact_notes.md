# PR 5b measurements so far (M1 Pro, load 3 to 6)

Harness ($SP/harness, C, 4M tokens, best of 5), ns per lookup, padded (fse_words) path:
- first D1 (while loop over smaller tags, load 2/3): hits8 22 ns vs master 5; probe dominated, scaled with load (1/2: 15, 1/4: 10): branch mispredictions of the variable loop.
- count of smaller tags over 4 slots (branch free): hits8 8.2.
- + padded tail without branch, two-load compare 9..16 B, hash without loop for <= 8 B and one-multiply finish: hits8 7.35, real glove 8.85 vs 7.0.
- load 1/2: hits8 6.4, real glove 7.9 vs 7.3, googlenews 13.4 vs 11.1.
- rejected: parallel compare of 4 slots (ternary 12.9; arithmetic masks 7.0..19.9 worse), power-of-two home with shift (no change), count over 2 slots (no change), D3 32-byte slots inline 16 (hits8 7.4..7.9, real not better).
- padded hash alone: new 2.2..2.8 vs master 2.0..2.5 ns: the gap is the probe (second dependent load), +1 ns short hits, +3 ns 9..16 B hits (arena line).

bench_vocab (extension, 1 round, load 1/2, specialized padded calls):
glove: real@fse_words 1.057, real external 0.904, hits8 0.924, hits16 0.928, cold16 0.910, miss 0.547, pieces 0.847; nbytes 14.1 vs 36.5 MB; build 0.014 vs 0.016 s.
googlenews: real@fse_words 1.135, real external 0.999, hits8 0.921, hits16 1.066, cold16 1.397, miss 0.779, pieces 0.960; nbytes 132.7 vs 309.7 MB; build 0.13 vs 0.17 s.
forcing fse_words onto the unpadded loads: no better (glove 1.12).

bench_runtime throughput float32 (synthetic w{i} words), 2 paired runs: branch/base within noise (0.95..1.10).
