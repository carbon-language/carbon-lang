// RUN: clang-cc -fsyntax-only -verify %s
template <typename T> struct S {
  S() { }
  S(T t);
};

template struct S<int>;

void f() {
  S<int> s1;
  S<int> s2(10);
}
