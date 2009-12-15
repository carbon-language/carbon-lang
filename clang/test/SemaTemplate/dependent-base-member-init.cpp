// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR4381
template<class T> struct X {};
template<typename T> struct Y : public X<T>::X { };

// PR4621
class A1 {
  A1(int x) {}
};
template<class C> class B1 : public A1 {
  B1(C x) : A1(x.x) {}
};
class A2 { A2(int x, int y); };
template <class C> class B2 {
  A2 x;
  B2(C x) : x(x.x, x.y) {}
};
template <class C> class B3 {
  C x;
  B3() : x(1,2) {}
};

// PR4627
template<typename _Container> class insert_iterator {
    _Container* container;
    insert_iterator(_Container& __x) : container(&__x) {}
};

// PR4763
template<typename T> struct s0 {};
template<typename T> struct s0_traits {};
template<typename T> struct s1 : s0<typename s0_traits<T>::t0> {
  s1() {}
};

