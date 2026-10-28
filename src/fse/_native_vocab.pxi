# Included by _native.pyx: the vocabulary table, on the C core in _native_vocab.h.

from libc.stdint cimport int64_t, uint8_t, uint32_t, uint64_t, uintptr_t

cdef uINT_t NOT_FOUND = <uINT_t>0xFFFFFFFF

# Return codes of the native routines, translated into exceptions with the GIL.
cdef enum:
    NATIVE_OK = 0
    NATIVE_NO_MEMORY = -1
    NATIVE_BAD_UTF8 = -2
    NATIVE_BAD_ROW = -3


cdef extern from "_native_vocab.h":
    ctypedef struct fse_vocab_slot:
        pass
    ctypedef struct fse_vocab:
        const fse_vocab_slot *slots
        const uint8_t *arena
        uint64_t size
        uint64_t seed
    ctypedef struct fse_vocab_words:
        const uint64_t *words
        const int64_t *offsets
        const uint8_t *arena
    ctypedef struct fse_vocab_plan:
        int attempt
        int64_t max_shift, last, second, first
    int64_t FSE_VOCAB_MAX_WORDS, FSE_VOCAB_MAX_WORD_BYTES, FSE_VOCAB_MAX_ARENA
    int64_t FSE_VOCAB_MAX_SHIFT, FSE_VOCAB_TAIL_SLOTS
    int FSE_VOCAB_SEEDS, FSE_VOCAB_OK, FSE_VOCAB_NO_MEMORY, FSE_VOCAB_DUPLICATE
    const uint64_t *fse_vocab_seeds
    uint32_t fse_vocab_find(const fse_vocab *v, const uint8_t *p, int64_t n, int padded) nogil
    uint64_t fse_vocab_hash(uint64_t seed, const uint8_t *p, int64_t n, int padded,
                            uint64_t *key) nogil
    uint32_t fse_vocab_tag(uint64_t h, int64_t n) nogil
    int64_t fse_vocab_measure(const int64_t *offsets, int64_t n, int64_t *too_long,
                              int64_t *arena_bytes) nogil
    void fse_vocab_copy(const uint8_t *bytes, const int64_t *offsets, int64_t n, uint8_t *arena,
                        uint64_t *words) nogil
    int fse_vocab_plan_build(const fse_vocab_words *w, int64_t n, uint64_t size, int64_t max_shift,
                             uint64_t *keys, uint64_t *tmp, fse_vocab_plan *plan) nogil
    void fse_vocab_fill(fse_vocab_slot *slots, int64_t n_slots, const fse_vocab_words *w,
                        const uint64_t *keys, int64_t n, uint64_t size) nogil

# The states of a VocabTable: a table is built once, and only a ready one is read.
cdef enum:
    TABLE_NEW = 0
    TABLE_BUILDING = 1
    TABLE_READY = 2
    TABLE_FAILED = 3

cdef enum:
    _SLOT_BYTES = 16
    _ALIGN = 64


def _vocab_tag(word, int attempt=0):
    """The tag of word (bytes) under the seed of an attempt; for tests."""
    if not 0 <= attempt < FSE_VOCAB_SEEDS:
        raise ValueError(f"attempt must be between 0 and {FSE_VOCAB_SEEDS - 1}")
    cdef bytes encoded = <bytes?>word
    cdef const unsigned char *p = encoded
    cdef uint64_t key
    cdef uint64_t h = fse_vocab_hash(fse_vocab_seeds[attempt], p, len(word), 0, &key)
    return fse_vocab_tag(h, len(word))


cdef object _checked_offsets(vocab_bytes, vocab_offsets):
    """An own copy of the offsets, checked against the bytes: arrays the caller still
    holds (or a writable base of a read-only view) could change while it is read."""
    if (
        not isinstance(vocab_bytes, np.ndarray) or vocab_bytes.dtype != np.uint8
        or vocab_bytes.ndim != 1 or not vocab_bytes.flags.c_contiguous
    ):
        raise ValueError("vocab_bytes must be a C-contiguous uint8 vector")
    if (
        not isinstance(vocab_offsets, np.ndarray) or vocab_offsets.dtype != np.int64
        or vocab_offsets.ndim != 1 or not vocab_offsets.flags.c_contiguous
    ):
        raise ValueError("vocab_offsets must be a C-contiguous int64 vector")
    offsets = vocab_offsets.copy()
    if len(offsets) == 0 or offsets[0] != 0 or offsets[len(offsets) - 1] != len(vocab_bytes):
        raise ValueError(f"vocab_offsets must run from 0 to {len(vocab_bytes)}, the number of bytes")
    if len(offsets) - 1 >= FSE_VOCAB_MAX_WORDS:
        raise ValueError(f"a vocabulary holds fewer than 2**31 words, not {len(offsets) - 1}")
    return offsets


cdef class VocabTable:
    """Immutable map from the UTF-8 bytes of a word to its vocabulary index.

    Built without the GIL from the vocabulary of word vectors: ``vocab_bytes`` (uint8)
    holds the words one after another, word i is ``vocab_bytes[vocab_offsets[i]:
    vocab_offsets[i + 1]]``. An ordered open-addressing table of 16-byte slots, at most
    2/3 full (_native_vocab.h): a word of at most 8 bytes is stored in its slot, a longer
    one in the table's arena, which holds only those. The table keeps its own copies of
    the words; it holds fewer than 2**31 words of fewer than 2**24 bytes each, and the
    words of more than 8 bytes hold fewer than 2**40 bytes.
    """

    cdef object _slots
    cdef object _arena
    cdef Py_ssize_t n_words
    cdef fse_vocab view
    cdef int _state
    cdef readonly int64_t _max_shift
    cdef readonly int _attempt

    def __init__(self, vocab_bytes, vocab_offsets, *, _max_shift_limit=None):
        if self._state != TABLE_NEW:
            raise RuntimeError("a VocabTable is built once; make a new one")
        if _max_shift_limit is not None and (
            isinstance(_max_shift_limit, bool) or not isinstance(_max_shift_limit, int) or _max_shift_limit < 0
        ):
            raise ValueError("_max_shift_limit must be a non-negative int")
        self._state = TABLE_BUILDING
        try:
            self._build(vocab_bytes, vocab_offsets,
                        FSE_VOCAB_MAX_SHIFT if _max_shift_limit is None else _max_shift_limit)
        except BaseException:
            self._state = TABLE_FAILED
            raise
        self._state = TABLE_READY

    cdef _build(self, vocab_bytes, vocab_offsets, int64_t limit):
        offsets_array = _checked_offsets(vocab_bytes, vocab_offsets)
        cdef const int64_t[::1] offsets = offsets_array
        cdef int64_t n = len(offsets) - 1
        cdef int64_t too_long, arena_bytes, bad, n_slots
        cdef fse_vocab_slot *slots = NULL
        cdef np.ndarray slot_array
        cdef uintptr_t address
        with nogil:
            bad = fse_vocab_measure(&offsets[0], n, &too_long, &arena_bytes)
        if bad >= 0:
            raise ValueError("vocab_offsets must not decrease")
        if too_long >= 0:
            raise ValueError(f"word {too_long} has 2**24 or more bytes")
        if arena_bytes >= FSE_VOCAB_MAX_ARENA:
            raise ValueError("the words of more than 8 bytes hold 2**40 or more bytes")
        self._arena = np.empty(max(arena_bytes, 1), dtype=np.uint8)
        cdef uint64_t size = <uint64_t>(2 * n + 1)
        cdef fse_vocab_words w
        cdef fse_vocab_plan plan
        cdef uint64_t *words = <uint64_t *>malloc(max(n, 1) * sizeof(uint64_t))
        cdef uint64_t *keys = <uint64_t *>malloc(max(n, 1) * sizeof(uint64_t))
        cdef uint64_t *tmp = <uint64_t *>malloc(max(n, 1) * sizeof(uint64_t))
        cdef int code = FSE_VOCAB_NO_MEMORY
        cdef const uint8_t *data = <const uint8_t *>np.PyArray_DATA(vocab_bytes)
        cdef uint8_t *arena = <uint8_t *>np.PyArray_DATA(self._arena)
        try:
            if words != NULL and keys != NULL and tmp != NULL:
                w.words, w.offsets, w.arena = words, &offsets[0], arena
                with nogil:
                    fse_vocab_copy(data, &offsets[0], n, arena, words)
                    code = fse_vocab_plan_build(&w, n, size, limit, keys, tmp, &plan)
            free(tmp)
            tmp = NULL
            self._check_plan(code, &plan, limit)
            n_slots = max(<int64_t>size, plan.last + 1) + FSE_VOCAB_TAIL_SLOTS
            slot_array = np.empty(n_slots * _SLOT_BYTES + _ALIGN, dtype=np.uint8)
            self._slots = slot_array
            address = <uintptr_t>np.PyArray_DATA(slot_array)
            slots = <fse_vocab_slot *>((address + _ALIGN - 1) & ~<uintptr_t>(_ALIGN - 1))
            with nogil:
                fse_vocab_fill(slots, n_slots, &w, keys, n, size)
        finally:
            free(words)
            free(keys)
            free(tmp)
        self.n_words = n
        self._max_shift, self._attempt = plan.max_shift, plan.attempt
        self.view.slots, self.view.arena = slots, arena
        self.view.size, self.view.seed = size, fse_vocab_seeds[plan.attempt]

    cdef _check_plan(self, int code, const fse_vocab_plan *plan, int64_t limit):
        if code == FSE_VOCAB_NO_MEMORY:
            raise MemoryError("no memory for the vocabulary table")
        if code == FSE_VOCAB_DUPLICATE:
            raise ValueError(f"word {plan.second} is also word {plan.first} of the vocabulary")
        if code != FSE_VOCAB_OK:
            raise ValueError(
                f"no seed places every word within {limit} slots of its home; the "
                "vocabulary looks crafted to collide"
            )

    cdef inline int _check_ready(self) except -1:
        if self._state != TABLE_READY:
            raise RuntimeError("this VocabTable was not built; make a new one")
        return 0

    @property
    def nbytes(self):
        """The bytes the table holds: its slots and its arena of the longer words."""
        self._check_ready()
        return self._slots.nbytes + self._arena.nbytes

    def __reduce__(self):
        raise TypeError("a VocabTable cannot be pickled; build it from the word vectors")

    def __len__(self):
        self._check_ready()
        return self.n_words

    def get(self, word):
        """Vocabulary index of word (str or bytes), or None; for tests."""
        self._check_ready()
        encoded = word if isinstance(word, bytes) else (<str?>word).encode("utf-8")
        cdef uINT_t index = fse_vocab_find(&self.view, encoded, len(encoded), 0)
        return None if index == NOT_FOUND else index
