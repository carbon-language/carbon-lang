// RUN: clang-cc -fsyntax-only -verify %s

template<typename T>
class X {
public:
  struct C { T &foo(); };

  struct D {
    struct E { T &bar(); };
    struct F; // expected-note{{member is declared here}}
  };
};

X<int>::C *c1;
X<float>::C *c2;

X<int>::X *xi;
X<float>::X *xf;

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
  f->foo(); // expected-error{{implicit instantiation of undefined member 'struct X<float>::D::F'}}
  
}
