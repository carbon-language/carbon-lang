// RUN: clang-cc -fsyntax-only -verify %s

// PR5057
namespace test0 {
  namespace std {
    class X {
    public:
      template<typename T> friend struct Y;
    };
  }

  namespace std {
    template<typename T> struct Y {};
  }
}

namespace test1 {
  template<typename T> void f1(T) { } // expected-note{{here}}

  class X {
    template<typename T> friend void f0(T);
    template<typename T> friend void f1(T);
  };

  template<typename T> void f0(T) { }
  template<typename T> void f1(T) { } // expected-error{{redefinition}}
}

// PR4768
namespace test2 {
  template<typename T> struct X0 {
    template<typename U> friend struct X0;
  };
  
  template<typename T> struct X0<T*> {
    template<typename U> friend struct X0;
  };

  template<> struct X0<int> {
    template<typename U> friend struct X0;
  };

  template<typename T> struct X1 {
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
}

// PR5332
namespace test3 {
  template <typename T> class Foo {
    template <typename U>
    friend class Foo;
  };

  Foo<int> foo;

  template<typename T, T Value> struct X2a;

  template<typename T, int Size> struct X2b;

  template<typename T>
  class X3 {
    template<typename U, U Value> friend struct X2a;
    template<typename U, T Value> friend struct X2b;
  };

  X3<int> x3i; // okay

  X3<long> x3l; // FIXME: should cause an instantiation-time failure
}

// PR5716
namespace test4 {
  template<typename> struct A {
    template<typename T> friend void f(const A<T>&);
  };

  template<typename T> void f(const A<T>&) {
    int a[sizeof(T) ? -1 : -1]; // expected-error {{array size is negative}}
  }

  void f() {
    f(A<int>()); // expected-note {{in instantiation of function template specialization}}
  }
}
