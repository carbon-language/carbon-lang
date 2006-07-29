// RUN: clang -E -C %s | grep '^// foo$' &&
// RUN: clang -E -C %s | grep -F '^/* bar */$'

// foo
/* bar */


