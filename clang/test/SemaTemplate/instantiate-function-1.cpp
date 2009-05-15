// RUN: clang-cc -fsyntax-only -verify %s
template<typename T, typename U>
struct X0 {
  void f(T x, U y) { 
    x + y; // expected-error{{invalid operands}}
  }
};

struct X1 { };

template struct X0<int, float>;
template struct X0<int*, int>;
template struct X0<int X1::*, int>; // expected-note{{instantiation of}}

template<typename T>
struct X2 {
  void f(T);

  T g(T x, T y) {
    /* DeclStmt */;
    T *xp = &x, &yr = y; // expected-error{{pointer to a reference}}
    /* NullStmt */;
  }
};

template struct X2<int>;
template struct X2<int&>; // expected-note{{instantiation of}}

template<typename T>
struct X3 {
  void f(T) {
    Label:
    T x;
    goto Label;
  }
};

template struct X3<int>;

template <typename T> struct X4 {
  T f() const {
    return; // expected-warning{{non-void function 'f' should return a value}}
  }
  
  T g() const {
    return 1; // expected-warning{{void function 'g' should not return a value}}
  }
};

template struct X4<void>; // expected-note{{in instantiation of template class 'X4<void>' requested here}}
template struct X4<int>; // expected-note{{in instantiation of template class 'X4<int>' requested here}}

struct Incomplete; // expected-note{{forward declaration}}

template<typename T> struct X5 {
  T f() { } // expected-error{{incomplete result type}}
};
void test_X5(X5<Incomplete> x5); // okay!

template struct X5<Incomplete>; // expected-note{{instantiation}}

template<typename T, typename U, typename V> struct X6 {
  U f(T t, U u, V v) {
    // IfStmt
    if (t > 0)
      return u;
    else
      return v; // expected-error{{incompatible type}}
  }
};

struct ConvertibleToInt {
  operator int() const;
};

template struct X6<ConvertibleToInt, float, char>;
template struct X6<bool, int, int*>; // expected-note{{instantiation}}

template <typename T> struct X7 {
  void f() {
    void *v = this;
  }
};

template struct X7<int>;
