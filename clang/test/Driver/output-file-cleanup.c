// RUN: touch %t.o
// RUN: not %clang -c -DCRASH -o %t.o -MMD -MF %t.d %s
// RUN: test ! -f %t.o
// RUN: test ! -f %t.d

// RUN: touch %t.o
// RUN: not %clang -c -DMISSING -o %t.o -MMD -MF %t.d %s
// RUN: test ! -f %t.o
// RUN: test ! -f %t.d

// RUN: touch %t.o
// RUN: not %clang -c -o %t.o -MMD -MF %t.d %s
// RUN: test ! -f %t.o
// RUN: test -f %t.d

// FIXME: %t.o is not touched with -no-integrated-as.
// XFAIL: mingw32,ppc
// REQUIRES: shell
// REQUIRES: crash-recovery

#ifdef CRASH
#pragma clang __debug crash
#elif defined(MISSING)
#include "nonexistent.h"
#else
invalid C code
#endif

// RUN: touch %t1.c
// RUN: echo "invalid C code" > %t2.c
// RUN: cd %T && not %clang -c %t1.c %t2.c
// RUN: test -f %t1.o
// RUN: test ! -f %t2.o

// RUN: touch %t1.c
// RUN: touch %t2.c
// RUN: chmod -r %t2.c
// RUN: cd %T && not %clang -c %t1.c %t2.c
// RUN: test -f %t1.o
// RUN: test ! -f %t2.o
