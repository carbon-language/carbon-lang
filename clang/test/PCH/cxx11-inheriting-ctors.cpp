// RUN: %clang_cc1 -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -include-pch %t -verify %s

// expected-no-diagnostics

#ifndef HEADER_INCLUDED
#define HEADER_INCLUDED

struct Base {
  Base(int) {}

  template <typename T>
  Base(T) {}
};

struct Test : Base {
  using Base::Base;
};

template <typename T>
struct Test2 : Base {
  using Base::Base;
};

template <typename B>
struct Test3 : B {
  using B::B;
};

#else

Test test1a(42);
Test test1b(nullptr);
Test2<int> test2a(42);
Test2<int> test2b(nullptr);
Test3<Base> test3a(42);
Test3<Base> test3b(nullptr);

#endif // HEADER_INCLUDED
