// RUN: clang-cc -fsyntax-only -verify %s
template<typename T>
void call_f0(T x) {
  x.Base::f0();
}

struct Base {
  void f0();
};

struct X0 : Base { 
  typedef Base CrazyBase;
};

void test_f0(X0 x0) {
  call_f0(x0);
}

template<typename TheBase, typename T>
void call_f0_through_typedef(T x) {
  typedef TheBase Base2;
  x.Base2::f0();
}

void test_f0_through_typedef(X0 x0) {
  call_f0_through_typedef<Base>(x0);
}

template<typename TheBase, typename T>
void call_f0_through_typedef2(T x) {
  typedef TheBase CrazyBase; // expected-note{{current scope}}
  x.CrazyBase::f0(); // expected-error{{ambiguous}} \
                     // expected-error 2{{no member named}}
}

struct OtherBase { };

struct X1 : Base, OtherBase { 
  typedef OtherBase CrazyBase; // expected-note{{object type}}
};

void test_f0_through_typedef2(X0 x0, X1 x1) {
  call_f0_through_typedef2<Base>(x0);
  call_f0_through_typedef2<OtherBase>(x1); // expected-note{{instantiation}}
  call_f0_through_typedef2<Base>(x1); // expected-note{{instantiation}}
}


