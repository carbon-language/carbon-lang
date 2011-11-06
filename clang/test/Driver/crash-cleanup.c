// RUN: not %clang -o %t.o -MMD -MF %t.d %s
// RUN: test ! -f %t.o
// RUN: test ! -f %t.d
// REQUIRES: shell
// REQUIRES: crash-recovery

// FIXME: Failing since r143846 (original commit), needs to be investigated.
// XFAIL: *

#pragma clang __debug crash
