// RUN: clang --emit-llvm -o %t %s &&
// RUN: grep -e "alias" %t
// XFAIL

static inline int foo () { return 0; }
int bar () __attribute__ ((alias ("foo")));
