// RUN: clang-tidy %s -checks=-*,misc-move-constructor-init -- -std=c++14 | FileCheck %s -implicit-check-not="{{warning|error}}:"

template <class T> struct remove_reference      {typedef T type;};
template <class T> struct remove_reference<T&>  {typedef T type;};
template <class T> struct remove_reference<T&&> {typedef T type;};

template <typename T>
typename remove_reference<T>::type&& move(T&& arg) {
  return static_cast<typename remove_reference<T>::type&&>(arg);
}

struct C {
  C() = default;
  C(const C&) = default;
};

struct B {
  B() {}
  B(const B&) {}
  B(B &&) {}
};

struct D : B {
  D() : B() {}
  D(const D &RHS) : B(RHS) {}
  // CHECK: :[[@LINE+3]]:16: warning: move constructor initializes base class by calling a copy constructor [misc-move-constructor-init]
  // CHECK: 19:3: note: copy constructor being called
  // CHECK: 20:3: note: candidate move constructor here
  D(D &&RHS) : B(RHS) {}
};

struct E : B {
  E() : B() {}
  E(const E &RHS) : B(RHS) {}
  E(E &&RHS) : B(move(RHS)) {} // ok
};

struct F {
  C M;

  F(F &&) : M(C()) {} // ok
};

struct G {
  G() = default;
  G(const G&) = default;
  G(G&&) = delete;
};

struct H : G {
  H() = default;
  H(const H&) = default;
  H(H &&RHS) : G(RHS) {} // ok
};

struct I {
  I(const I &) = default; // suppresses move constructor creation
};

struct J : I {
  J(J &&RHS) : I(RHS) {} // ok
};

struct K {}; // Has implicit copy and move constructors, is trivially copyable
struct L : K {
  L(L &&RHS) : K(RHS) {} // ok
};

struct M {
  B Mem;
  // CHECK: :[[@LINE+1]]:16: warning: move constructor initializes class member by calling a copy constructor [misc-move-constructor-init]
  M(M &&RHS) : Mem(RHS.Mem) {}
};

struct N {
  B Mem;
  N(N &&RHS) : Mem(move(RHS.Mem)) {}
};
