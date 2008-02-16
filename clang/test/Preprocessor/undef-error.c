// RUN: not clang %s -pedantic-errors -E
// PR2045

#define b
#undef a b
