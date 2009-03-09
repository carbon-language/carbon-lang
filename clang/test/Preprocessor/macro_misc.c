// RUN: clang %s -E 2>&1 | not grep error

// This should not be rejected.
#ifdef defined
#endif

