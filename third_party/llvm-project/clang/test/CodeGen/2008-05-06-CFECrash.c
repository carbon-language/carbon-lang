// RUN: %clang_cc1 -emit-llvm -O2 %s -o /dev/null
// PR2292.
__inline__ __attribute__ ((__pure__)) int g (void) {}
void f (int k) { k = g (); }
