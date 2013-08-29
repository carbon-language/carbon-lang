// RUN: %clang_cc1 -emit-pch -o %t.1 %s
// RUN: %clang_cc1 -error-on-deserialized-decl S1_keyfunc -include-pch %t.1 -emit-pch -o %t.2 %s
// RUN: %clang_cc1 -error-on-deserialized-decl S1_method -include-pch %t.2 -emit-llvm-only %s

#ifndef HEADER1
#define HEADER1
// Header.

struct S1 {
  void S1_method();
  virtual void S1_keyfunc();
};

#elif !defined(HEADER2)
#define HEADER2

// Chained PCH.
S1 *p;

#else

// Using the headers.

void test(S1*) {
}

#endif
