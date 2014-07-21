// RUN: %clang_cc1 -std=c++11 -verify %s

// Note that this puts the expected lines before the directives to work around
// limitations in the -verify mode.

void test(int *List, int Length) {
  int i = 0;

#pragma unroll
  while (i + 1 < Length) {
    List[i] = i;
  }

#pragma unroll 4
  while (i - 1 < Length) {
    List[i] = i;
  }

#pragma unroll(8)
  while (i - 2 < Length) {
    List[i] = i;
  }

#pragma unroll
#pragma unroll(8)
  while (i - 3 < Length) {
    List[i] = i;
  }

#pragma clang loop unroll(enable)
#pragma unroll(8)
  while (i - 4 < Length) {
    List[i] = i;
  }

#pragma unroll
#pragma clang loop unroll_count(4)
  while (i - 5 < Length) {
    List[i] = i;
  }

/* expected-error {{expected ')'}} */ #pragma unroll(4
/* expected-error {{missing argument to '#pragma unroll'; expected a positive integer value}} */ #pragma unroll()
/* expected-warning {{extra tokens at end of '#pragma unroll'}} */ #pragma unroll 1 2
  while (i-6 < Length) {
    List[i] = i;
  }

/* expected-error {{invalid argument; expected a positive integer value}} */ #pragma unroll(()
/* expected-error {{invalid argument; expected a positive integer value}} */ #pragma unroll -
/* expected-error {{invalid argument; expected a positive integer value}} */ #pragma unroll(0)
/* expected-error {{invalid argument; expected a positive integer value}} */ #pragma unroll 0
/* expected-error {{invalid argument; expected a positive integer value}} */ #pragma unroll(3000000000)
/* expected-error {{invalid argument; expected a positive integer value}} */ #pragma unroll 3000000000
  while (i-8 < Length) {
    List[i] = i;
  }

#pragma unroll
/* expected-error {{expected a for, while, or do-while loop to follow '#pragma unroll'}} */ int j = Length;
#pragma unroll 4
/* expected-error {{expected a for, while, or do-while loop to follow '#pragma unroll'}} */ int k = Length;

/* expected-error {{incompatible directives 'unroll(disable)' and '#pragma unroll(4)'}} */ #pragma unroll 4
#pragma clang loop unroll(disable)
  while (i-10 < Length) {
    List[i] = i;
  }

/* expected-error {{duplicate directives '#pragma unroll' and '#pragma unroll'}} */ #pragma unroll
#pragma unroll
  while (i-14 < Length) {
    List[i] = i;
  }

/* expected-error {{duplicate directives 'unroll(enable)' and '#pragma unroll'}} */ #pragma unroll
#pragma clang loop unroll(enable)
  while (i-15 < Length) {
    List[i] = i;
  }

/* expected-error {{duplicate directives '#pragma unroll(4)' and '#pragma unroll(4)'}} */ #pragma unroll 4
#pragma unroll(4)
  while (i-16 < Length) {
    List[i] = i;
  }

#pragma unroll
/* expected-error {{expected statement}} */ }
