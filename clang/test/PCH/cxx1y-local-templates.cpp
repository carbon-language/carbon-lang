// RUN: %clang_cc1 -pedantic-errors -std=c++1y -emit-pch %s -o %t-cxx1y
// RUN: %clang_cc1 -ast-print -pedantic-errors -std=c++1y -include-pch %t-cxx1y  %s | FileCheck -check-prefix=CHECK-PRINT %s

#ifndef HEADER_INCLUDED

#define HEADER_INCLUDED

auto nested_local_call_all() {
  struct Inner1 {
    auto inner1_foo(char c) {
      struct Inner2 {
        template<class T> T inner2_foo(T t) {
          return t;
        }
      };
      return Inner2{};
    }
  };
  return Inner1{}.inner1_foo('a').inner2_foo(4);
}


auto nested_local() {
  struct Inner1 {
    auto inner1_foo(char c) {
      struct Inner2 {
        template<class T> T inner2_foo(T t) {
          return t;
        }
      };
      return Inner2{};
    }
  };
  return Inner1{};
}


int test() {
  auto A = nested_local_call_all();
  auto B = nested_local();
  auto C = B.inner1_foo('a');
  C.inner2_foo(3.14);

}


#else

// CHECK-PRINT: int nested_local_call_all
// CHECK-PRINT: nested_local
auto nested_local_call_all();

int test(int y) {
  return nested_local_call_all();
}


#endif
