// RUN: %clang_cc1 -triple i386-apple-darwin9 -fsyntax-only -verify %s

int x __attribute__((aligned(3))); // expected-error {{requested alignment is not a power of 2}}

// PR3254
short g0[3] __attribute__((aligned));
short g0_chk[__alignof__(g0) == 16 ? 1 : -1]; 

// <rdar://problem/6840045>
typedef char ueber_aligned_char __attribute__((aligned(8)));

struct struct_with_ueber_char {
  ueber_aligned_char c;
};

char c = 0;

char a0[__alignof__(ueber_aligned_char) == 8? 1 : -1] = { 0 };
char a1[__alignof__(struct struct_with_ueber_char) == 8? 1 : -1] = { 0 };
char a2[__alignof__(c) == 1? : -1] = { 0 };
char a3[sizeof(c) == 1? : -1] = { 0 };
