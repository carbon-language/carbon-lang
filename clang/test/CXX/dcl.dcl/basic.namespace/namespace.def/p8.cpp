// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

// Fun things you can do with inline namespaces:

inline namespace X {
  void f1();

  inline namespace Y {
    void f2();

    template <typename T> class C {};
  }

  // Specialize and partially specialize somewhere else.
  template <> class C<int> {};
  template <typename T> class C<T*> {};
}

// Qualified and unqualified lookup as if member of enclosing NS.
void foo1() {
  f1();
  ::f1();
  X::f1();
  Y::f1(); // expected-error {{no member named 'f1' in namespace 'X::Y'}}

  f2();
  ::f2();
  X::f2();
  Y::f2();
}

template <> class C<float> {};
template <typename T> class C<T&> {};

template class C<double>;


// As well as all the fun with ADL.

namespace ADL {
  struct Outer {};

  inline namespace IL {
    struct Inner {};

    void fo(Outer);
  }

  void fi(Inner);

  inline namespace IL2 {
    void fi2(Inner);
  }
}

void foo2() {
  ADL::Outer o;
  ADL::Inner i;
  fo(o);
  fi(i);
  fi2(i);
}

// Let's not forget overload sets.
struct Distinct {};
inline namespace Over {
  void over(Distinct);
}
void over(int);

void foo3() {
  Distinct d;
  ::over(d);
}
