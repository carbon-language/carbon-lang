// RUN: clang-cc %s -Eonly 2>&1 | grep error
// RUN: clang-cc %s -Eonly 2>&1 | not grep unterminated
// RUN: clang-cc %s -Eonly 2>&1 | not grep scratch

#define COMM / ## *
COMM

