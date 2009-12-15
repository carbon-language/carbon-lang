// RUN: %clang_cc1 %s -Eonly 2>&1 | grep error
// RUN: %clang_cc1 %s -Eonly 2>&1 | not grep unterminated
// RUN: %clang_cc1 %s -Eonly 2>&1 | not grep scratch

#define COMM / ## *
COMM

