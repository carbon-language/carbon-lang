// RUN: clang-cc -E -C %s | grep '^// foo$' &&
// RUN: clang-cc -E -C %s | grep -F '^/* bar */$'

// foo
/* bar */


