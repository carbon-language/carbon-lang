// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s


// If P is an rvalue reference to a cv-unqualified template parameter
// and the argument is an lvalue, the type "lvalue reference to A" is
// used in place of A for type deduction.
template<typename T> struct X { };

template<typename T> X<T> f0(T&&);

struct Y { };

template<typename T> T prvalue();
template<typename T> T&& xvalue();
template<typename T> T& lvalue();

void test_f0() {
  X<int> xi0 = f0(prvalue<int>());
  X<int> xi1 = f0(xvalue<int>());
  X<int&> xi2 = f0(lvalue<int>());
  X<Y> xy0 = f0(prvalue<Y>());
  X<Y> xy1 = f0(xvalue<Y>());
  X<Y&> xy2 = f0(lvalue<Y>());
}
