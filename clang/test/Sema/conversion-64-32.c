// RUN: %clang_cc1 -fsyntax-only -verify -Wshorten-64-to-32 -triple x86_64-apple-darwin %s

int test0(long v) {
  return v; // expected-warning {{implicit conversion loses integer precision}}
}


// rdar://9546171
typedef int  int4  __attribute__ ((vector_size(16)));
typedef long long long2 __attribute__((__vector_size__(16)));

int4 test1(long2 a) {
  int4  v127 = a;  // no warning.
  return v127; 
}

// <rdar://problem/10759934>
// Don't warn about -Wshorten-64-to-32 in unreachable code.
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
int rdar10759934() {
  uint32_t thing = 0;
  uint64_t thing2 = 0;

  switch (sizeof(thing2)) {
  case 8:
    break;
  case 4:
    thing = thing2; // no-warning
  default:
    break;
  }

  return 0;
}
