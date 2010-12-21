// RUN: %clang_cc1 -fsyntax-only -verify %s
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

    // FIXME: the redeclaration note ends up here because redeclaration
    // lookup ends up finding the friend target from X3<int>.
    template<typename U, T Value> friend struct X2b; // expected-error {{template non-type parameter has a different type 'long' in template redeclaration}} \
      // expected-note {{previous non-type template parameter with type 'int' is here}}
  };

  X3<int> x3i; // okay

  X3<long> x3l; // expected-note {{in instantiation}}
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

namespace test5 {
  class outer {
    class foo;
    template <typename T> friend struct cache;
  };
  class outer::foo {
    template <typename T> friend struct cache;
  };
}

// PR6022
namespace PR6022 {
  template <class T1, class T2 , class T3  > class A;

  namespace inner {
    template<class T1, class T2, class T3, class T> 
    A<T1, T2, T3>& f0(A<T1, T2, T3>&, T);
  } 

  template<class T1, class T2, class T3>
  class A {
    template<class U1, class U2, class U3, class T>  
    friend A<U1, U2, U3>& inner::f0(A<U1, U2, U3>&, T);
  };
}

namespace FriendTemplateDefinition {
  template<unsigned > struct int_c { };

  template<typename T>
  struct X {
    template<unsigned N>
    friend void f(X, int_c<N>) {
      int value = N;
    };
  };

  void test_X(X<int> x, int_c<5> i5) {
    f(x, i5);
  }
}

namespace PR7013a {
  template<class > struct X0
  {
    typedef int type;
  };
  template<typename > struct X1
  {
  };
  template<typename , typename T> struct X2
  {
    typename T::type e;
  };
  namespace N
  {
    template <typename = int, typename = X1<int> > struct X3
    {
      template <typename T1, typename T2, typename B> friend void op(X2<T1, T2>& , B);
    };
    template <typename Ch, typename Tr, typename B> void op(X2<Ch, Tr>& , B)
    {
      X2<int, Tr> s;
    }
  }
  int n()
  {
    X2<int, X0<int> > ngs;
    N::X3<> b;
    op(ngs, b);
    return 0;
  }
}

namespace PR7013b {
  template<class > struct X0
  {
    typedef int type;
  };
  template<typename > struct X1
  {
  };
  template<typename , typename T> struct X2
  {
    typename T::type e;
  };
  namespace N
  {
    template <typename = X1<int> > struct X3
    {
      template <typename T1, typename T2, typename B> friend void op(X2<T1, T2>& , B);
    };
    template <typename Ch, typename Tr, typename B> void op(X2<Ch, Tr>& , B)
    {
      X2<int, Tr> s;
    }
  }
  int n()
  {
    X2<int, X0<int> > ngs;
    N::X3<> b;
    op(ngs, b);
    return 0;
  }

}

namespace PR8649 {
  template<typename T, typename U, unsigned N>
  struct X {
    template<unsigned M> friend class X<T, U, M>; // expected-error{{partial specialization cannot be declared as a friend}}
  };

  X<int, float, 7> x;
}
