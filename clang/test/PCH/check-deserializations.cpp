// RUN: %clang_cc1 -emit-pch -o %t.1 %s
// RUN: %clang_cc1 -error-on-deserialized-decl S1_keyfunc -error-on-deserialized-decl S3 -include-pch %t.1 -emit-pch -o %t.2 %s
// RUN: %clang_cc1 -error-on-deserialized-decl S1_method -error-on-deserialized-decl S3 -include-pch %t.2 -emit-llvm-only %s

#ifndef HEADER1
#define HEADER1
// Header.

struct S1 {
  void S1_method();
  virtual void S1_keyfunc();
};

struct S3 {};

struct S2 {
  operator S3();
};

#elif !defined(HEADER2)
#define HEADER2

// Chained PCH.
S1 *s1;
S2 *s2;

#else

// Using the headers.

void test(S1*, S2*) {
}

#endif
