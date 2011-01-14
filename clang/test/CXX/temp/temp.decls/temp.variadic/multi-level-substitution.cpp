// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

template<typename T, T ...Values> struct value_tuple {};

template<typename T>
struct X0 {
  template<T ...Values>
  void f(value_tuple<T, Values...> * = 0);
};

void test_X0() {
  X0<int>().f<1, 2, 3, 4, 5>();
}

namespace PacksAtDifferentLevels {
  template<typename...> struct tuple { };
  template<typename T, typename U> struct pair { };

  template<typename ...Types>
  struct X {
    template<typename> struct Inner;

    template<typename ...YTypes>
    struct Inner<tuple<pair<Types, YTypes>...> > {
      static const unsigned zero = sizeof...(Types) - sizeof...(YTypes);
    };
  };

  int check0[X<short, int, long>::Inner<tuple<pair<short, unsigned short>,
                                             pair<int, unsigned int>,
                                             pair<long, unsigned long>>
                                       >::zero == 0? 1 : -1];
}
