// RUN: clang-cc -fsyntax-only -verify %s

enum Enum { val = 1 };
template <Enum v> struct C {
  typedef C<v> Self;
};
template struct C<val>;

template<typename T>
struct get_size {
  static const unsigned value = sizeof(T);
};

template<typename T>
struct X0 {
  enum {
    Val1 = get_size<T>::value,
    Val2,
    SumOfValues = Val1 + Val2
  };
};

X0<int> x0i;
