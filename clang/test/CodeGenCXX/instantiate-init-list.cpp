// RUN: clang-cc %s -emit-llvm-only -verify

struct F {
  void (*x)();
};
void G();
template<class T> class A {
  A();
};
template<class T> A<T>::A() {
  static F f = { G };
}
A<int> a;
