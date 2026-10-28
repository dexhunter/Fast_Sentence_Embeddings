/* The vocabulary table of fse: the map from the UTF-8 bytes of a word to its index
   (ARCHITECTURE.md, "Text sources").

   An ordered open-addressing table. Every word has a 32-bit tag, 27 bits of its hash
   above a 5-bit class of its length (0 to 30 exact, 31 for longer words), and a home
   slot that grows with the tag. Sorting the words by tag therefore lays the table out
   in Robin Hood order, and a lookup stops at the first slot with a greater tag. Equal
   tags mean equal length classes, so a word of at most 8 bytes, stored in its slot,
   compares exactly; a longer word is compared in the table's arena, which holds only the
   longer words. The table is at most 2/3 full and ends in at least 4 empty slots, whose
   tag is greater than every word's, so no lookup wraps around or reads past the end.

   The hash is no persistence format: a table is built when it is used and never saved.
   It absorbs a word 8 bytes at a time and takes the length only at the end, so the hash
   of every prefix of a word can be finished from the state of its last full block. */
#ifndef FSE_NATIVE_VOCAB_H
#define FSE_NATIVE_VOCAB_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* The keys load words as little-endian integers in two ways that must agree. */
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#error "fse supports little-endian platforms only"
#endif

#define FSE_VOCAB_INLINE_BYTES 8
#define FSE_VOCAB_NOT_FOUND UINT32_C(0xFFFFFFFF)
#define FSE_VOCAB_EMPTY_TAG UINT32_C(0xFFFFFFFF)
#define FSE_VOCAB_MAX_WORDS (INT64_C(1) << 31)      /* a table holds fewer words */
#define FSE_VOCAB_MAX_WORD_BYTES (INT64_C(1) << 24) /* a word has fewer bytes */
#define FSE_VOCAB_MAX_ARENA (INT64_C(1) << 40)      /* the longer words have fewer bytes */
#define FSE_VOCAB_MAX_SHIFT 256 /* the furthest a word may lie from its home */
#define FSE_VOCAB_TAIL_SLOTS 4 /* empty slots past the last word and past every home */
#define FSE_VOCAB_SEEDS 4
#define FSE_VOCAB_RUN_PAIRS 16 /* runs of equal tags up to this are compared pairwise */

#define FSE_VOCAB_OK 0
#define FSE_VOCAB_NO_MEMORY (-1)
#define FSE_VOCAB_DUPLICATE (-2)
#define FSE_VOCAB_CLUSTERED (-3)

/* A slot: the tag, the index and the word itself (zero-padded, at most 8 bytes) or the
   offset of the word in the arena << 24 | its length. An empty slot has the tag
   FSE_VOCAB_EMPTY_TAG and the index FSE_VOCAB_NOT_FOUND. */
typedef struct {
    uint32_t tag;
    uint32_t id;
    uint64_t word;
} fse_vocab_slot;

/* What a lookup reads; filled once when the table is built, then read by any thread. */
typedef struct {
    const fse_vocab_slot *slots;
    const uint8_t *arena;
    uint64_t size; /* the homes are 0 to size - 1 */
    uint64_t seed;
} fse_vocab;

/* A build tries the seeds in this order until no word lies too far from its home. */
static const uint64_t fse_vocab_seeds[FSE_VOCAB_SEEDS] = {
    UINT64_C(0x9E3779B97F4A7C15), UINT64_C(0xD1B54A32D192ED03),
    UINT64_C(0x8CB92BA72F3D8DD7), UINT64_C(0xA0761D6478BD642F),
};

static inline uint64_t fse_vocab_load64(const uint8_t *p) {
    uint64_t w;
    memcpy(&w, p, 8);
    return w;
}

static inline uint64_t fse_vocab_load32(const uint8_t *p) {
    uint32_t w;
    memcpy(&w, p, 4);
    return w;
}

/* The last block of the n bytes at p, its 1 to 8 bytes from offset 8 * ((n - 1) / 8), as
   a little-endian word, zero above them; 0 for n = 0. With padded, the 8 bytes from the
   start of that block are readable; without, only p[0:n] is read, with overlapping
   loads that agree on the bytes they share. */
static inline uint64_t fse_vocab_last(const uint8_t *p, int64_t n, int padded) {
    if (n == 0) return 0;
    unsigned k = (unsigned)((n - 1) & 7) + 1;
    const uint8_t *t = p + (n - (int64_t)k);
    if (padded) return fse_vocab_load64(t) & (~UINT64_C(0) >> (64 - 8 * k));
    if (n >= 8) return fse_vocab_load64(p + n - 8) >> (8 * (8 - k));
    if (k >= 4) return fse_vocab_load32(t) | (fse_vocab_load32(t + k - 4) << (8 * (k - 4)));
    return (uint64_t)t[0] | ((uint64_t)t[k >> 1] << (8 * (k >> 1))) | ((uint64_t)t[k - 1] << (8 * (k - 1)));
}

static inline uint64_t fse_vocab_absorb(uint64_t s, uint64_t w) {
    s = (s ^ w) * UINT64_C(0xBF58476D1CE4E5B9);
    return s ^ (s >> 29);
}

/* The length, then one multiply, whose top bits (home and tag) depend on every bit. */
static inline uint64_t fse_vocab_finish(uint64_t s, int64_t n) {
    return (s ^ (uint64_t)n * UINT64_C(0x9E3779B97F4A7C15)) * UINT64_C(0x94D049BB133111EB);
}

/* The hash of the n bytes at p: their blocks of 8 bytes absorbed in turn, the last one
   zero-padded, then the length. key receives the word as a slot stores it if n <= 8.
   padded as for fse_vocab_last; both give the same hash and key. */
static inline uint64_t fse_vocab_hash(uint64_t seed, const uint8_t *p, int64_t n, int padded,
                                      uint64_t *key) {
    uint64_t s = seed;
    int64_t b = 0;
    for (; b + 8 < n; b += 8) s = fse_vocab_absorb(s, fse_vocab_load64(p + b));
    *key = fse_vocab_last(p, n, padded);
    return fse_vocab_finish(fse_vocab_absorb(s, *key), n);
}

static inline uint32_t fse_vocab_tag(uint64_t h, int64_t n) {
    uint32_t t = ((uint32_t)(h >> 32) & ~UINT32_C(31)) | (uint32_t)(n < 31 ? n : 31);
    return t == FSE_VOCAB_EMPTY_TAG ? t - 1 : t; /* class 31 becomes 30, both long */
}

/* Monotone in the tag, below size for size <= 2**32. */
static inline uint64_t fse_vocab_home(uint32_t tag, uint64_t size) {
    return ((uint64_t)tag * size) >> 32;
}

/* Whether the n > 8 bytes at a and p are equal; up to 16 bytes in two overlapping loads
   without a call. */
static inline int fse_vocab_equal_long(const uint8_t *a, const uint8_t *p, int64_t n) {
    if (n <= 16)
        return ((fse_vocab_load64(a) ^ fse_vocab_load64(p)) |
                (fse_vocab_load64(a + n - 8) ^ fse_vocab_load64(p + n - 8))) == 0;
    return memcmp(a, p, (size_t)n) == 0;
}

/* The index of the n bytes at p, with their tag and key, or FSE_VOCAB_NOT_FOUND.

   From the home, the slots hold the words displaced from earlier homes (smaller tags),
   then those of this tag, then greater tags or an empty slot: the smaller tags are a
   prefix. Counting them 4 slots at a time, without a branch per slot, finds the first
   candidate; a loop over their varying number would mispredict. */
static inline uint32_t fse_vocab_find_hashed(const fse_vocab *v, uint32_t tag, uint64_t key,
                                             const uint8_t *p, int64_t n) {
    const fse_vocab_slot *s = v->slots + fse_vocab_home(tag, v->size);
    for (;;) {
        unsigned c = (unsigned)(s[0].tag < tag) + (s[1].tag < tag) + (s[2].tag < tag) + (s[3].tag < tag);
        s += c;
        if (c < 4) break;
    }
    for (; s->tag == tag; s++) {
        if (n <= FSE_VOCAB_INLINE_BYTES
                ? s->word == key
                : (s->word & 0xFFFFFF) == (uint64_t)n &&
                      fse_vocab_equal_long(v->arena + (s->word >> 24), p, n))
            return s->id;
    }
    return FSE_VOCAB_NOT_FOUND; /* an empty slot or a greater tag */
}

static inline uint32_t fse_vocab_find(const fse_vocab *v, const uint8_t *p, int64_t n, int padded) {
    uint64_t key;
    uint64_t h = fse_vocab_hash(v->seed, p, n, padded, &key);
    return fse_vocab_find_hashed(v, fse_vocab_tag(h, n), key, p, n);
}

/* The build ---------------------------------------------------------------------- */

/* The words of a table being built, in its own copies: words[i] as a slot stores it,
   the lengths from offsets. */
typedef struct {
    const uint64_t *words;
    const int64_t *offsets;
    const uint8_t *arena;
} fse_vocab_words;

/* Of a valid offsets vector: the first word with too many bytes (or -1) and the bytes of
   the words longer than 8 bytes. Returns the first i with offsets[i + 1] < offsets[i],
   or -1. */
static int64_t fse_vocab_measure(const int64_t *offsets, int64_t n, int64_t *too_long,
                                 int64_t *arena_bytes) {
    *too_long = -1;
    *arena_bytes = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t len = offsets[i + 1] - offsets[i];
        if (len < 0) return i;
        if (len >= FSE_VOCAB_MAX_WORD_BYTES && *too_long < 0) *too_long = i;
        if (len > FSE_VOCAB_INLINE_BYTES) *arena_bytes += len;
    }
    return -1;
}

/* Read every word once from the caller's bytes into words and arena. */
static void fse_vocab_copy(const uint8_t *bytes, const int64_t *offsets, int64_t n,
                           uint8_t *arena, uint64_t *words) {
    uint64_t pos = 0;
    for (int64_t i = 0; i < n; i++) {
        const uint8_t *p = bytes + offsets[i];
        uint64_t len = (uint64_t)(offsets[i + 1] - offsets[i]);
        if (len <= FSE_VOCAB_INLINE_BYTES) {
            uint64_t key = 0;
            memcpy(&key, p, (size_t)len);
            words[i] = key;
        } else {
            memcpy(arena + pos, p, (size_t)len);
            words[i] = pos << 24 | len;
            pos += len;
        }
    }
}

static inline int64_t fse_vocab_length(const fse_vocab_words *w, int64_t i) {
    return w->offsets[i + 1] - w->offsets[i];
}

/* keys[i] = tag << 32 | i under seed, hashed from the own copies. */
static void fse_vocab_keys(const fse_vocab_words *w, int64_t n, uint64_t seed, uint64_t *keys) {
    for (int64_t i = 0; i < n; i++) {
        int64_t len = fse_vocab_length(w, i);
        uint64_t key, padded[2] = {w->words[i], 0}; /* 8 readable bytes past the word */
        uint64_t h = len <= FSE_VOCAB_INLINE_BYTES
                         ? fse_vocab_hash(seed, (const uint8_t *)padded, len, 1, &key)
                         : fse_vocab_hash(seed, w->arena + (w->words[i] >> 24), len, 0, &key);
        keys[i] = (uint64_t)fse_vocab_tag(h, len) << 32 | (uint64_t)i;
    }
}

/* Stable LSD radix sort of the keys by their tag, 8 bits per pass; O(n) for any input. */
static void fse_vocab_sort(uint64_t *keys, uint64_t *tmp, int64_t n) {
    uint64_t *from = keys, *to = tmp;
    for (int shift = 32; shift < 64; shift += 8) {
        int64_t count[257] = {0};
        for (int64_t i = 0; i < n; i++) count[((from[i] >> shift) & 255) + 1]++;
        for (int b = 0; b < 256; b++) count[b + 1] += count[b];
        for (int64_t i = 0; i < n; i++) to[count[(from[i] >> shift) & 255]++] = from[i];
        uint64_t *t = from;
        from = to;
        to = t;
    }
    /* four passes: the result is back in keys */
}

/* Orders two words of one run of equal tags by length, then bytes; 0 if they are equal. */
static int fse_vocab_compare(const fse_vocab_words *w, uint32_t a, uint32_t b) {
    int64_t la = fse_vocab_length(w, a), lb = fse_vocab_length(w, b);
    if (la != lb) return la < lb ? -1 : 1;
    if (la <= FSE_VOCAB_INLINE_BYTES)
        return w->words[a] == w->words[b] ? 0 : w->words[a] < w->words[b] ? -1 : 1;
    return memcmp(w->arena + (w->words[a] >> 24), w->arena + (w->words[b] >> 24), (size_t)la);
}

static inline void fse_vocab_candidate(int64_t i, int64_t j, int64_t *second, int64_t *first) {
    if (i < *second) {
        *second = i;
        *first = j;
    }
}

static void fse_vocab_run_pairs(const fse_vocab_words *w, const uint64_t *keys, int64_t r,
                                int64_t *second, int64_t *first) {
    for (int64_t x = 0; x < r; x++)
        for (int64_t y = x + 1; y < r; y++) {
            uint32_t a = (uint32_t)keys[x], b = (uint32_t)keys[y];
            if (fse_vocab_compare(w, a, b) == 0)
                fse_vocab_candidate(a > b ? a : b, a < b ? a : b, second, first);
        }
}

/* Bottom-up merge sort of r ids by (word, id), O(r log r) for any input. */
static void fse_vocab_merge_sort(const fse_vocab_words *w, uint32_t *ids, uint32_t *tmp, int64_t r) {
    uint32_t *from = ids, *to = tmp;
    for (int64_t width = 1; width < r; width *= 2) {
        for (int64_t lo = 0; lo < r; lo += 2 * width) {
            int64_t mid = lo + width < r ? lo + width : r, hi = lo + 2 * width < r ? lo + 2 * width : r;
            int64_t x = lo, y = mid, k = lo;
            while (x < mid && y < hi) {
                int c = fse_vocab_compare(w, from[x], from[y]);
                to[k++] = (c < 0 || (c == 0 && from[x] < from[y])) ? from[x++] : from[y++];
            }
            while (x < mid) to[k++] = from[x++];
            while (y < hi) to[k++] = from[y++];
        }
        uint32_t *t = from;
        from = to;
        to = t;
    }
    if (from != ids) memcpy(ids, from, (size_t)r * sizeof(uint32_t));
}

/* A long run, sorted: equal words are adjacent and ordered by id. */
static int fse_vocab_run_sorted(const fse_vocab_words *w, const uint64_t *keys, int64_t r,
                                int64_t *second, int64_t *first) {
    uint32_t *ids = (uint32_t *)malloc((size_t)(2 * r) * sizeof(uint32_t));
    if (!ids) return FSE_VOCAB_NO_MEMORY;
    for (int64_t x = 0; x < r; x++) ids[x] = (uint32_t)keys[x];
    fse_vocab_merge_sort(w, ids, ids + r, r);
    int64_t start = 0;
    for (int64_t x = 1; x < r; x++) {
        if (fse_vocab_compare(w, ids[x - 1], ids[x]) != 0)
            start = x;
        else
            fse_vocab_candidate(ids[x], ids[start], second, first);
    }
    free(ids);
    return FSE_VOCAB_OK;
}

/* Of the keys sorted by tag (equal words have equal tags): the smallest second
   occurrence of a word, *second, and its earlier twin, *first; *second is n if every
   word is distinct. */
static int fse_vocab_duplicate(const fse_vocab_words *w, const uint64_t *keys, int64_t n,
                               int64_t *second, int64_t *first) {
    *second = n;
    *first = -1;
    for (int64_t a = 0; a < n;) {
        int64_t b = a + 1;
        while (b < n && (keys[b] >> 32) == (keys[a] >> 32)) b++;
        if (b - a > FSE_VOCAB_RUN_PAIRS) {
            if (fse_vocab_run_sorted(w, keys + a, b - a, second, first)) return FSE_VOCAB_NO_MEMORY;
        } else if (b - a > 1) {
            fse_vocab_run_pairs(w, keys + a, b - a, second, first);
        }
        a = b;
    }
    return FSE_VOCAB_OK;
}

/* The furthest any of the sorted keys lies from its home; *last receives the last
   position. Position i is max(home_i, position_{i-1} + 1). */
static int64_t fse_vocab_layout(const uint64_t *keys, int64_t n, uint64_t size, int64_t *last) {
    int64_t pos = -1, shift = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t home = (int64_t)fse_vocab_home((uint32_t)(keys[i] >> 32), size);
        pos = home > pos + 1 ? home : pos + 1;
        if (pos - home > shift) shift = pos - home;
    }
    *last = pos;
    return shift;
}

/* The outcome of planning a table: its seed and where its words lie. */
typedef struct {
    int attempt;
    int64_t max_shift, last, second, first;
} fse_vocab_plan;

/* Hash, sort and lay out the words under the seeds in turn, until none lies more than
   max_shift slots from its home; keys then hold the sorted keys of that seed. */
static int fse_vocab_plan_build(const fse_vocab_words *w, int64_t n, uint64_t size,
                                int64_t max_shift, uint64_t *keys, uint64_t *tmp,
                                fse_vocab_plan *plan) {
    for (int attempt = 0; attempt < FSE_VOCAB_SEEDS; attempt++) {
        fse_vocab_keys(w, n, fse_vocab_seeds[attempt], keys);
        fse_vocab_sort(keys, tmp, n);
        if (attempt == 0) {
            if (fse_vocab_duplicate(w, keys, n, &plan->second, &plan->first)) return FSE_VOCAB_NO_MEMORY;
            if (plan->second < n) return FSE_VOCAB_DUPLICATE;
        }
        plan->attempt = attempt;
        plan->max_shift = fse_vocab_layout(keys, n, size, &plan->last);
        if (plan->max_shift <= max_shift) return FSE_VOCAB_OK;
    }
    return FSE_VOCAB_CLUSTERED;
}

/* Write the sorted keys into n_slots slots, all others empty. */
static void fse_vocab_fill(fse_vocab_slot *slots, int64_t n_slots, const fse_vocab_words *w,
                           const uint64_t *keys, int64_t n, uint64_t size) {
    for (int64_t j = 0; j < n_slots; j++) {
        slots[j].tag = FSE_VOCAB_EMPTY_TAG;
        slots[j].id = FSE_VOCAB_NOT_FOUND;
        slots[j].word = 0;
    }
    int64_t pos = -1;
    for (int64_t i = 0; i < n; i++) {
        uint32_t tag = (uint32_t)(keys[i] >> 32), id = (uint32_t)keys[i];
        int64_t home = (int64_t)fse_vocab_home(tag, size);
        pos = home > pos + 1 ? home : pos + 1;
        slots[pos].tag = tag;
        slots[pos].id = id;
        slots[pos].word = w->words[id];
    }
}

#endif /* FSE_NATIVE_VOCAB_H */
