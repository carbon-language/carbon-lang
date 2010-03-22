// RUN: %clang -triple=i386-apple-darwin10 -msse3 -fsyntax-only -verify -std=c89 %s

// PR6658
#include <xmmintrin.h>

