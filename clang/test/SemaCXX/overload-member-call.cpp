// RUN: %clang_cc1 -fsyntax-only -verify %s

struct X {
  int& f(int) const; // expected-note 2 {{candidate function}}
  float& f(int); // expected-note 2 {{candidate function}}

  void test_f(int x) const {
    int& i = f(x);
  }

  void test_f2(int x) {
    float& f2 = f(x);
  }

  int& g(int) const; // expected-note 2 {{candidate function}}
  float& g(int); // expected-note 2 {{candidate function}}
  static double& g(double); // expected-note 2 {{candidate function}}

  void h(int);

  void test_member() {
    float& f1 = f(0);
    float& f2 = g(0);
    double& d1 = g(0.0);
  }

  void test_member_const() const {
    int &i1 = f(0);
    int &i2 = g(0);
    double& d1 = g(0.0);
  }

  static void test_member_static() {
    double& d1 = g(0.0);
    g(0); // expected-error{{call to 'g' is ambiguous; candidates are:}}
  }
};

void test(X x, const X xc, X* xp, const X* xcp, volatile X xv, volatile X* xvp) {
  int& i1 = xc.f(0);
  int& i2 = xcp->f(0);
  float& f1 = x.f(0);
  float& f2 = xp->f(0);
  xv.f(0); // expected-error{{no matching member function for call to 'f'; candidates are:}}
  xvp->f(0); // expected-error{{no matching member function for call to 'f'; candidates are:}}

  int& i3 = xc.g(0);
  int& i4 = xcp->g(0);
  float& f3 = x.g(0);
  float& f4 = xp->g(0);
  double& d1 = xp->g(0.0);
  double& d2 = X::g(0.0);
  X::g(0); // expected-error{{call to 'g' is ambiguous; candidates are:}}
  
  X::h(0); // expected-error{{call to non-static member function without an object argument}}
}

struct X1 {
  int& member();
  float& member() const;
};

struct X2 : X1 { };

void test_X2(X2 *x2p, const X2 *cx2p) {
  int &ir = x2p->member();
  float &fr = cx2p->member();
}
