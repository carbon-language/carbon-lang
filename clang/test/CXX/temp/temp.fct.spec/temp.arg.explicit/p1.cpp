// RUN: clang-cc -fsyntax-only %s

template<typename T> struct A { };

template<typename T> T make(A<T>);

void test_make() {
  int& ir0 = make<int&>(A<int&>());
  A<int> a0 = make< A<int> >(A<A<int> >());
}

