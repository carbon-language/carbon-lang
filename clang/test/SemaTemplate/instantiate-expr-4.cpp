// RUN: clang-cc -fsyntax-only -verify %s

// ---------------------------------------------------------------------
// C++ Functional Casts
// ---------------------------------------------------------------------
template<int N>
struct ValueInit0 {
  int f() {
    return int();
  }
};

template struct ValueInit0<5>;

template<int N>
struct FunctionalCast0 {
  int f() {
    return int(N);
  }
};

template struct FunctionalCast0<5>;

struct X {
  X(int, int);
};

template<int N, int M>
struct BuildTemporary0 {
  X f() {
    return X(N, M);
  }
};

template struct BuildTemporary0<5, 7>;
