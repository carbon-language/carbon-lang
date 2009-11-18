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

// PR4655
template<class C> struct A {};
template <> struct A<int>{A(const A<int>&);};
struct B { A<int> x; B(B& a) : x(a.x) {} };

struct X2 {
  X2();
  X2(X2&);	// expected-note {{candidate function}}
  template<typename T> X2(T);
};

X2 test(bool Cond, X2 x2) {
  if (Cond)
    return x2; // okay, uses copy constructor
  
  return X2(); // expected-error{{no viable conversion from 'struct X2' to 'struct X2' is possible}}
}

struct X3 {
  template<typename T> X3(T);
};

template<> X3::X3(X3); // expected-error{{must pass its first argument by reference}}

struct X4 {
  X4();
  ~X4();
  X4(X4&);	// expected-note {{candidate function}}
  template<typename T> X4(const T&, int = 17);
};

X4 test_X4(bool Cond, X4 x4) {
  X4 a(x4, 17); // okay, constructor template
  X4 b(x4); // okay, copy constructor
  return X4(); // expected-error{{no viable conversion}}
}
