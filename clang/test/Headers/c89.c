// RUN: %clang -ccc-host-triple i386-apple-darwin10 -fsyntax-only -Xclang -verify -std=c89 %s

// PR6658
#include <xmmintrin.h>

