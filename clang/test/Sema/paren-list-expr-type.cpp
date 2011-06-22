// RUN: %clang -cc1 -ast-dump %s | not grep NULL
// Makes sure that we don't introduce null types when handling
// ParenListExpr.

template<typename T> class X { void f() { X x(*this); } };

template<typename T> class Y { Y() : t(1) {} T t; };

template<typename T> class Z { Z() : b(true) {} const bool b; };

template<typename T> class A : public Z<T> { A() : Z<T>() {} };

class C {};
template<typename T> class D : public C { D(): C() {} };

void f() { (int)(1, 2); }

