// RUN: clang %s -Eonly -verify

// This should not be rejected.
#ifdef defined
#endif

