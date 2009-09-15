// RUN: clang-cc -fsyntax-only -verify %s

struct X0 { // expected-note{{candidate}}
  X0(int); // expected-note{{candidate}}
  template<typename T> X0(T);
  template<typename T, typename U> X0(T*, U*);
  
  // PR4761
  template<typename T> X0() : f0(T::foo) {}
  int f0;
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

template<typename T>
struct X1 {
  X1(const X1&);
  template<typename U> X1(const X1<U>&);
};

template<typename T>
struct Outer {
  typedef X1<T> A;
  
  A alloc;
  
  explicit Outer(const A& a) : alloc(a) { }
};

void test_X1(X1<int> xi) {
  Outer<int> oi(xi);
  Outer<float> of(xi);
}
