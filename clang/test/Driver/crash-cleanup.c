// RUN: not %clang -o %t.o -MMD -MF %t.d %s
// RUN: test ! -f %t.o
// RUN: test ! -f %t.d
// REQUIRES: shell
// REQUIRES: crash-recovery

// XFAIL: *

#pragma clang __debug crash
