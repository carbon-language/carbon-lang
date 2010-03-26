// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
class X {
public:
  struct C { T &foo(); };

  struct D {
    struct E { T &bar(); }; // expected-error{{cannot form a reference to 'void'}}
    struct F; // expected-note{{member is declared here}}
  };
};

X<int>::C *c1;
X<float>::C *c2;

X<int>::X *xi; // expected-error{{qualified reference to 'X' is a constructor name rather than a type wherever a constructor can be declared}}
X<float>::X *xf; // expected-error{{qualified reference to 'X' is a constructor name rather than a type wherever a constructor can be declared}}

void test_naming() {
  c1 = c2; // expected-error{{incompatible type assigning 'X<float>::C *', expected 'X<int>::C *'}}
  xi = xf;  // expected-error{{incompatible type assigning}}
    // FIXME: error above doesn't print the type X<int>::X cleanly!
}

void test_instantiation(X<double>::C *x,
                        X<float>::D::E *e,
                        X<float>::D::F *f) {
  double &dr = x->foo();
  float &fr = e->bar();
  f->foo(); // expected-error{{implicit instantiation of undefined member 'X<float>::D::F'}}
  
}


X<void>::C *c3; // okay
X<void>::D::E *e1; // okay
X<void>::D::E e2; // expected-note{{in instantiation of member class 'X<void>::D::E' requested here}}

// Redeclarations.
namespace test1 {
  template <typename T> struct Registry {
    class node;
    static node *Head;
    class node {
      node(int v) { Head = this; }
    };
  };
  void test() {
    Registry<int>::node node(0);
  }
}

// Redeclarations during explicit instantiations.
namespace test2 {
  template <typename T> class A {
    class Foo;
    class Foo {
      int foo();
    };
  };
  template class A<int>;

  template <typename T> class B {
    class Foo;
    class Foo {
      typedef int X;
    };
    typename Foo::X x;
    class Foo;
  };
  template class B<int>;

  template <typename T> class C {
    class Foo;
    class Foo;
  };
  template <typename T> class C<T>::Foo {
    int x;
  };
  template class C<int>;
}
