// RUN: clang-cc -fsyntax-only -verify %s

// PR5057
namespace std {
  class X {
  public:
    template<typename T>
    friend struct Y;
  };
}

namespace std {
  template<typename T>
  struct Y
  {
  };
}


namespace N {
  template<typename T> void f1(T) { } // expected-note{{here}}

  class X {
    template<typename T> friend void f0(T);
    template<typename T> friend void f1(T);
  };

  template<typename T> void f0(T) { }
  template<typename T> void f1(T) { } // expected-error{{redefinition}}
}

// PR4768
template<typename T>
struct X0 {
  template<typename U> friend struct X0;
};

template<typename T>
struct X0<T*> {
  template<typename U> friend struct X0;
};

template<>
struct X0<int> {
  template<typename U> friend struct X0;
};

template<typename T>
struct X1 {
  template<typename U> friend void f2(U);
  template<typename U> friend void f3(U);
};

template<typename U> void f2(U);

X1<int> x1i;
X0<int*> x0ip;

template<> void f2(int);

// FIXME: Should this declaration of f3 be required for the specialization of
// f3<int> (further below) to work? GCC and EDG don't require it, we do...
template<typename U> void f3(U);

template<> void f3(int);

// PR5332
template <typename T>
class Foo {
  template <typename U>
  friend class Foo;
};

Foo<int> foo;
