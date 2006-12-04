// RUN: not clang %s -E &&
// RUN: clang %s -E -fno-operator-names

// Not valid in C++ unless -fno-operator-names is passed.
#define and foo


