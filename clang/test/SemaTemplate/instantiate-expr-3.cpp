// RUN: clang-cc -fsyntax-only -verify %s

// ---------------------------------------------------------------------
// Imaginary literals
// ---------------------------------------------------------------------
template<typename T>
struct ImaginaryLiteral0 {
  void f(T &x) {
    x = 3.0I; // expected-error{{incompatible type}}
  }
};

template struct ImaginaryLiteral0<_Complex float>;
template struct ImaginaryLiteral0<int*>; // expected-note{{instantiation}}

// ---------------------------------------------------------------------
// Compound assignment operator
// ---------------------------------------------------------------------
namespace N1 {
  struct X { };

  int& operator+=(X&, int); // expected-note{{candidate}}
}

namespace N2 {
  long& operator+=(N1::X&, long); // expected-note{{candidate}}

  template<typename T, typename U, typename Result>
  struct PlusEquals0 {
    void f(T t, U u) {
      Result r = t += u; // expected-error{{ambiguous}}
    }
  };
}

namespace N3 {
  struct Y : public N1::X {
    short& operator+=(long); // expected-note{{candidate}}
  };
}

template struct N2::PlusEquals0<N1::X, int, int&>;
template struct N2::PlusEquals0<N1::X, long, long&>;
template struct N2::PlusEquals0<N3::Y, long, short&>;
template struct N2::PlusEquals0<int, int, int&>;
template struct N2::PlusEquals0<N3::Y, int, short&>; // expected-note{{instantiation}}

// ---------------------------------------------------------------------
// Conditional operator
// ---------------------------------------------------------------------
template<typename T, typename U, typename Result>
struct Conditional0 {
  void f(T t, U u) {
    Result result = t? : u;
  }
};

template struct Conditional0<int, int, int>;

// ---------------------------------------------------------------------
// Statement expressions
// ---------------------------------------------------------------------
template<typename T>
struct StatementExpr0 {
  void f(T t) {
    (void)({ if (t) t = t + 17; }); // expected-error{{invalid}}
  }
};

template struct StatementExpr0<int>;
template struct StatementExpr0<N1::X>; // expected-note{{instantiation}}

// ---------------------------------------------------------------------
// __builtin_shufflevector
// ---------------------------------------------------------------------
typedef __attribute__(( ext_vector_type(2) )) double double2;

template<typename T, typename U, int N, int M>
struct ShuffleVector0 {
  void f(T t, U u, double2 a, double2 b) {
    (void)__builtin_shufflevector(t, u, N, M); // expected-error{{index}}
    (void)__builtin_shufflevector(a, b, N, M);
    (void)__builtin_shufflevector(a, b, 2, 1);
  }
};

template struct ShuffleVector0<double2, double2, 2, 1>;
template struct ShuffleVector0<double2, double2, 4, 3>; // expected-note{{instantiation}}

// ---------------------------------------------------------------------
// __builtin_choose_expr
// ---------------------------------------------------------------------
template<bool Cond, typename T, typename U, typename Result>
struct Choose0 {
  void f(T t, U u) {
    Result r = __builtin_choose_expr(Cond, t, u); // expected-error{{lvalue}}
  }
};

template struct Choose0<true, int, float, int&>;
template struct Choose0<false, int, float, float&>;
template struct Choose0<true, int, float, float&>; // expected-note{{instantiation}}
