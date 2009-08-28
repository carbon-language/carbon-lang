// RUN: clang-cc -fsyntax-only -verify %s

template<typename T>
struct X0 {
  template<typename U> T f0(U);
  template<typename U> U& f1(T*, U); // expected-error{{pointer to a reference}} \
                                     // expected-note{{candidate}}
};

X0<int> x0i;
X0<void> x0v;
X0<int&> x0ir; // expected-note{{instantiation}}

void test_X0(int *ip, double *dp) {
  X0<int> xi;
  int i1 = xi.f0(ip);
  double *&dpr = xi.f1(ip, dp);
  xi.f1(dp, dp); // expected-error{{no matching}}

  X0<void> xv;
  double *&dpr2 = xv.f1(ip, dp);
}

