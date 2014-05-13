// RUN: %clang_cc1 -std=c++11 -emit-pch -o %t.a %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -include-pch %t.a %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

inline void test(int i) {
  switch (i) {
    case 1:
      // Notice that the NullStmt has two attributes.
      [[clang::fallthrough]][[clang::fallthrough]];
    case 2:
      break;
  }
}

#else

void foo(void) {
  test(1);
}

#endif
