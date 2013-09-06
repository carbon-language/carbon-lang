// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only
// expected-no-diagnostics

typedef __attribute__((ext_vector_type(2)))  char char2;
typedef __attribute__((ext_vector_type(4)))  unsigned int uint4;
typedef __attribute__((ext_vector_type(8)))  long long8;

void vectorIncrementDecrementOps()
{
  char2 A = (char2)(1);
  uint4 B = (uint4)(1);
  long8 C = (long8)(1);

  A++;
  --A;
  B--;
  ++B;
  C++;
}
