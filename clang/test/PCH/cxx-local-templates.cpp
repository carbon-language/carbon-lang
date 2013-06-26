// RUN: %clang_cc1 -pedantic-errors -std=c++11 -emit-pch %s -o %t-cxx11
// RUN: %clang_cc1 -ast-print -pedantic-errors -std=c++11 -include-pch %t-cxx11  %s | FileCheck -check-prefix=CHECK-PRINT %s

#ifndef HEADER_INCLUDED

#define HEADER_INCLUDED

int nontemplate_test(double d) {
  struct Local {
    template<class T> T foo(T t) {
      return t;
    }
  };
  return Local{}.foo(d);
}

template<class U>
U template_test(U d) {
  struct Local {
    template<class T> T foo(T t) {
      return t;
    }
  };
  return Local{}.foo(d);
}

int nested_local() {
  struct Inner1 {
    int inner1_foo(char c) {
      struct Inner2 {
        template<class T> T inner2_foo(T t) {
          return t;
        }
      };
      return Inner2{}.inner2_foo(3.14);
    }
  };
  return Inner1{}.inner1_foo('a');
}

#else

// CHECK-PRINT: U template_test

// CHECK-PRINT: int nontemplate_test(double)

int nontemplate_test(double);

template double template_test(double);
int test2(int y) {
  return nontemplate_test(y) + template_test(y);
}


#endif
