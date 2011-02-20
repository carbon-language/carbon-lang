// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x

template<typename T>
struct only {
  only(T);
  template<typename U> only(U) = delete; // expected-note {{here}}
};

template<typename ...T>
void f(T ...t) {
  auto x(t...); // expected-error {{requires an initializer}} expected-error {{contains multiple expressions}}
  only<int> check = x;
}

void g() {
  f(); // expected-note {{here}}
  f(0);
  f(0, 1); // expected-note {{here}}
}


template<typename T>
bool h(T t) {
  auto a = t;
  decltype(a) b;
  a = a + b;

  auto p = new auto(t);

  only<double*> test = p; // expected-error {{conversion function from 'char *' to 'only<double *>'}}
  return p;
}

bool b = h('x'); // expected-note {{here}}
