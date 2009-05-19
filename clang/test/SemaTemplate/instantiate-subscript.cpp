// RUN: clang-cc -fsyntax-only -verify %s


struct Sub0 {
  int &operator[](int);
};

struct Sub1 {
  long &operator[](long);
};

struct ConvertibleToInt {
  operator int();
};

template<typename T, typename U, typename Result>
struct Subscript0 {
  void test(T t, U u) {
    Result &result = t[u]; // expected-error{{subscripted value is not}}
  }
};

template struct Subscript0<int*, int, int&>;
template struct Subscript0<Sub0, int, int&>;
template struct Subscript0<Sub1, ConvertibleToInt, long&>;
template struct Subscript0<Sub1, Sub0, long&>; // expected-note{{instantiation}}
