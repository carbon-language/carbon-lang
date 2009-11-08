// RUN: not clang-cc %s -E
// RUN: clang-cc %s -E -fno-operator-names

// Not valid in C++ unless -fno-operator-names is passed.
#define and foo


