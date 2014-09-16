// RUN: %clang_cc1 -verify -std=c++11 %s

template <typename T> struct OwnPtr {
  T *p;
  ~OwnPtr() {
    // expected-error@+1 {{invalid application of 'sizeof'}}
    static_assert(sizeof(T) > 0, "incomplete T");
    delete p;
  }
};

namespace use_vtable_for_vcall {
struct Incomplete; // expected-note {{forward declaration}}
struct A {
  virtual ~A() {}
  virtual void m() {}
};
struct B : A { // expected-note {{in instantiation}}
  B();
  virtual void m() { }
  virtual void m2() { static_cast<A *>(this)->m(); }
  OwnPtr<Incomplete> m_sqlError;
};

B *f() {
  return new B();
}
}

namespace dont_mark_qualified_vcall {
struct Incomplete;
struct A {
  virtual ~A() {}
  virtual void m() {}
};
struct B : A {
  B();
  // Previously we would mark B's vtable referenced to devirtualize this call to
  // A::m, even though it's not a virtual call.
  virtual void m() { A::m(); }
  OwnPtr<Incomplete> m_sqlError;
};

B *f() {
  return new B();
}
}
