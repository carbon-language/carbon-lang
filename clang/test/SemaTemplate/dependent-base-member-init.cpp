// RUN: clang-cc -fsyntax-only -verify %s

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

