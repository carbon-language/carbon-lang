// RUN: %clang_cc1 -std=c++11 -verify %s

// Note that this puts the expected lines before the directives to work around
// limitations in the -verify mode.

void test(int *List, int Length) {
  int i = 0;

#pragma clang loop vectorize(enable)
#pragma clang loop interleave(enable)
  while (i + 1 < Length) {
    List[i] = i;
  }

#pragma clang loop vectorize_width(4)
#pragma clang loop interleave_count(8)
  while (i < Length) {
    List[i] = i;
  }

#pragma clang loop vectorize(disable)
#pragma clang loop interleave(disable)
  while (i - 1 < Length) {
    List[i] = i;
  }

#pragma clang loop vectorize_width(4) interleave_count(8)
  while (i - 2 < Length) {
    List[i] = i;
  }

#pragma clang loop interleave_count(16)
  while (i - 3 < Length) {
    List[i] = i;
  }

  int VList[Length];
#pragma clang loop vectorize(disable) interleave(disable)
  for (int j : VList) {
    VList[j] = List[j];
  }

/* expected-error {{expected '('}} */ #pragma clang loop vectorize
/* expected-error {{expected '('}} */ #pragma clang loop interleave

/* expected-error {{expected ')'}} */ #pragma clang loop vectorize(enable
/* expected-error {{expected ')'}} */ #pragma clang loop interleave(enable

/* expected-error {{expected ')'}} */ #pragma clang loop vectorize_width(4
/* expected-error {{expected ')'}} */ #pragma clang loop interleave_count(4

/* expected-error {{missing option}} */ #pragma clang loop
/* expected-error {{invalid option 'badkeyword'}} */ #pragma clang loop badkeyword
/* expected-error {{invalid option 'badkeyword'}} */ #pragma clang loop badkeyword(enable)
/* expected-error {{invalid option 'badkeyword'}} */ #pragma clang loop vectorize(enable) badkeyword(4)
/* expected-warning {{extra tokens at end of '#pragma clang loop'}} */ #pragma clang loop vectorize(enable) ,

  while (i-4 < Length) {
    List[i] = i;
  }

/* expected-error {{invalid value 0; expected a positive integer value}} */ #pragma clang loop vectorize_width(0)
/* expected-error {{invalid value 0; expected a positive integer value}} */ #pragma clang loop interleave_count(0)
  while (i-5 < Length) {
    List[i] = i;
  }

/* expected-error {{invalid value -1294967296; expected a positive integer value}} */ #pragma clang loop vectorize_width(3000000000)
/* expected-error {{invalid value -1294967296; expected a positive integer value}} */ #pragma clang loop interleave_count(3000000000)
  while (i-6 < Length) {
    List[i] = i;
  }

/* expected-error {{missing value; expected a positive integer value}} */ #pragma clang loop vectorize_width(badvalue)
/* expected-error {{missing value; expected a positive integer value}} */ #pragma clang loop interleave_count(badvalue)
  while (i-6 < Length) {
    List[i] = i;
  }

/* expected-error {{invalid keyword 'badidentifier'; expected 'enable' or 'disable'}} */ #pragma clang loop vectorize(badidentifier)
/* expected-error {{invalid keyword 'badidentifier'; expected 'enable' or 'disable'}} */ #pragma clang loop interleave(badidentifier)
  while (i-7 < Length) {
    List[i] = i;
  }

#pragma clang loop vectorize(enable)
/* expected-error {{expected a for, while, or do-while loop to follow the '#pragma clang loop' directive}} */ int j = Length;
  List[0] = List[1];

  while (j-1 < Length) {
    List[j] = j;
  }

// FIXME: A bug in ParsedAttributes causes the order of the attributes to be
// processed in reverse. Consequently, the errors occur on the first of pragma
// of the next three tests rather than the last, and the order of the kinds
// is also reversed.

/* expected-error {{incompatible directives 'vectorize(disable)' and 'vectorize_width(4)'}} */ #pragma clang loop vectorize_width(4)
#pragma clang loop vectorize(disable)
/* expected-error {{incompatible directives 'interleave(disable)' and 'interleave_count(4)'}} */ #pragma clang loop interleave_count(4)
#pragma clang loop interleave(disable)
  while (i-8 < Length) {
    List[i] = i;
  }

/* expected-error {{duplicate directives 'vectorize(disable)' and 'vectorize(enable)'}} */ #pragma clang loop vectorize(enable)
#pragma clang loop vectorize(disable)
/* expected-error {{duplicate directives 'interleave(disable)' and 'interleave(enable)'}} */ #pragma clang loop interleave(enable)
#pragma clang loop interleave(disable)
  while (i-9 < Length) {
    List[i] = i;
  }

/* expected-error {{incompatible directives 'vectorize_width(4)' and 'vectorize(disable)'}} */ #pragma clang loop vectorize(disable)
#pragma clang loop vectorize_width(4)
/* expected-error {{incompatible directives 'interleave_count(4)' and 'interleave(disable)'}} */ #pragma clang loop interleave(disable)
#pragma clang loop interleave_count(4)
  while (i-10 < Length) {
    List[i] = i;
  }

/* expected-error {{duplicate directives 'vectorize_width(4)' and 'vectorize_width(8)'}} */ #pragma clang loop vectorize_width(8)
#pragma clang loop vectorize_width(4)
/* expected-error {{duplicate directives 'interleave_count(4)' and 'interleave_count(8)'}} */ #pragma clang loop interleave_count(8)
#pragma clang loop interleave_count(4)
  while (i-11 < Length) {
    List[i] = i;
  }

#pragma clang loop interleave(enable)
/* expected-error {{expected statement}} */ }
