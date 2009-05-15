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
    else { 
      if (t < 0)
        return v; // expected-error{{incompatible type}}
    }

    return v;
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

template<typename T> struct While0 {
  void f(T t) {
    while (t) {
    }

    while (T t2 = T()) ;
  }
};

template struct While0<float>;

template<typename T> struct Do0 {
  void f(T t) {
    do {
    } while (t); // expected-error{{not contextually}}
    
    do {
    } while (T t2 = T());
  }
};

struct NotConvertibleToBool { };
template struct Do0<ConvertibleToInt>;
template struct Do0<NotConvertibleToBool>; // expected-note{{instantiation}}

template<typename T> struct For0 {
  void f(T f, T l) {
    for (; f != l; ++f) {
      if (*f)
        continue;
      else if (*f == 17)
        break;
    }
  }
};

template struct For0<int*>;

template<typename T> struct Member0 {
  void f(T t) {
    t;
    t.f;
    t->f;
    
    T* tp;
    tp.f; // expected-error{{member reference base type 'T *' is not a structure or union}}
    tp->f;

    this->f;
    this.f; // expected-error{{member reference base type 'struct Member0 *const' is not a structure or union}}
  }
};
