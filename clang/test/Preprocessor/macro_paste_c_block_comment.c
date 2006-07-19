// RUN: clang %s -Eonly 2>&1 | grep error &&
// RUN: clang %s -Eonly 2>&1 | not grep unterminated &&
// RUN: clang %s -Eonly 2>&1 | not grep scratch

#define COMM / ## *
COMM

