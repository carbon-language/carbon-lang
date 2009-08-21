// RUN: clang-cc -fsyntax-only -verify %s

struct X0 { // expected-note{{candidate}}
  X0(int); // expected-note{{candidate}}
  template<typename T> X0(T);
  template<typename T, typename U> X0(T*, U*);
};

void accept_X0(X0);

void test_X0(int i, float f) {
  X0 x0a(i);
  X0 x0b(f);
  X0 x0c = i;
  X0 x0d = f;
  accept_X0(i);
  accept_X0(&i);
  accept_X0(f);
  accept_X0(&f);
  X0 x0e(&i, &f);
  X0 x0f(&f, &i);
  
  X0 x0g(f, &i); // expected-error{{no matching constructor}}
}
