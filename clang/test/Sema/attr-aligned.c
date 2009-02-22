// RUN: clang -triple i386-apple-darwin9 -fsyntax-only -verify %s

int x __attribute__((aligned(3))); // expected-error {{requested alignment is not a power of 2}}

// PR3254
short g0[3] __attribute__((aligned));
short g0_chk[__alignof__(g0) == 16 ? 1 : -1]; 

// PR3433
double g1;
short chk1[__alignof__(g1) == 8 ? 1 : -1]; 

