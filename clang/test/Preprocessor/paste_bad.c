// RUN: not clang -E %s
// GCC PR 20077

#define a a ## ##
#define a() a ## ##

