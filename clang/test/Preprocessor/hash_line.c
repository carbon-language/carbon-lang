// The 1 and # should not go on the same line.
// RUN: clang %s -E | not grep "1 #" &&
// RUN: clang %s -E | grep '^1$' &&
// RUN: clang %s -E | grep '^      #$'
1
#define EMPTY
EMPTY #

