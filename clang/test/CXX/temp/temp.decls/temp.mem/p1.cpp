// RUN: clang-cc -fsyntax-only -verify %s

template <class T> struct A {
  static T cond;
  
  template <class U> struct B {
    static T twice(U value) {
      return (cond ? value + value : value);
    }
  };
};

int foo() {
  A<bool>::cond = true;
  return A<bool>::B<int>::twice(4);
}
