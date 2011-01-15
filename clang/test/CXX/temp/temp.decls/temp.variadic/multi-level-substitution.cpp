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
    template<typename> struct Inner {
      static const unsigned value = 1;
    };

    template<typename ...YTypes>
    struct Inner<tuple<pair<Types, YTypes>...> > {
      static const unsigned value = sizeof...(Types) - sizeof...(YTypes);
    };
  };

  int check0[X<short, int, long>::Inner<tuple<pair<short, unsigned short>,
                                             pair<int, unsigned int>,
                                             pair<long, unsigned long>>
                                       >::value == 0? 1 : -1];

  int check1[X<short, int>::Inner<tuple<pair<short, unsigned short>,
                                        pair<int, unsigned int>,
                                        pair<long, unsigned long>>
                                       >::value == 1? 1 : -1]; 

  template<unsigned ...Values> struct unsigned_tuple { };
  template<typename ...Types>
  struct X1 {
    template<typename, typename> struct Inner {
      static const unsigned value = 0;
    };

    template<typename ...YTypes>
    struct Inner<tuple<pair<Types, YTypes>...>,
                 unsigned_tuple<sizeof(Types) + sizeof(YTypes)...>> {
      static const unsigned value = 1;
    };
  };

  int check2[X1<short, int, long>::Inner<tuple<pair<short, unsigned short>,
                                               pair<int, unsigned int>,
                                               pair<long, unsigned long>>,
                      unsigned_tuple<sizeof(short) + sizeof(unsigned short),
                                     sizeof(int) + sizeof(unsigned int),
                                     sizeof(long) + sizeof(unsigned long)>
                                       >::value == 1? 1 : -1];
  int check3[X1<short, int>::Inner<tuple<pair<short, unsigned short>,
                                         pair<int, unsigned int>,
                                         pair<long, unsigned long>>,
                      unsigned_tuple<sizeof(short) + sizeof(unsigned short),
                                     sizeof(int) + sizeof(unsigned int),
                                     sizeof(long) + sizeof(unsigned long)>
                                       >::value == 0? 1 : -1];

  template<typename ...Types>
  struct X2 {
    template<typename> struct Inner {
      static const unsigned value = 1;
    };

    template<typename R, typename ...YTypes>
    struct Inner<R(pair<Types, YTypes>...)> {
      static const unsigned value = sizeof...(Types) - sizeof...(YTypes);
    };
  };

  int check4[X2<short, int, long>::Inner<int(pair<short, unsigned short>,
                                            pair<int, unsigned int>,
                                            pair<long, unsigned long>)
                                     >::value == 0? 1 : -1];

  int check5[X2<short, int>::Inner<int(pair<short, unsigned short>,
                                       pair<int, unsigned int>,
                                       pair<long, unsigned long>)
                                     >::value == 1? 1 : -1]; 

  template<typename T, typename U>
  struct some_function_object {
    template<typename>
    struct result_of;
  };

  template<template<class> class...> struct metafun_tuple { };

  template<typename ...Types1>
  struct X3 {
    template<typename, typename> struct Inner {
      static const unsigned value = 0;
    };

    template<typename ...Types2>
    struct Inner<tuple<pair<Types1, Types2>...>,
                 metafun_tuple<some_function_object<Types1, Types2>::template result_of...> > {
      static const unsigned value = 1;
    };
  };

  int check6[X3<short, int, long>::Inner<tuple<pair<short, unsigned short>,
                                               pair<int, unsigned int>,
                                               pair<long, unsigned long>>,
                                 metafun_tuple<
                         some_function_object<short, unsigned short>::result_of,
                         some_function_object<int, unsigned int>::result_of,
                         some_function_object<long, unsigned long>::result_of>
                                     >::value == 1? 1 : -1];
  int check7[X3<short, int>::Inner<tuple<pair<short, unsigned short>,
                                               pair<int, unsigned int>,
                                               pair<long, unsigned long>>,
                                 metafun_tuple<
                         some_function_object<short, unsigned short>::result_of,
                         some_function_object<int, unsigned int>::result_of,
                         some_function_object<long, unsigned long>::result_of>
                                     >::value == 0? 1 : -1];

  template<unsigned I, unsigned J> struct unsigned_pair { };

  template<unsigned ...Values1>
  struct X4 {
    template<typename> struct Inner {
      static const unsigned value = 0;
    };

    template<unsigned ...Values2>
    struct Inner<tuple<unsigned_pair<Values1, Values2>...>> {
      static const unsigned value = 1;
    };
  };

  int check8[X4<1, 3, 5>::Inner<tuple<unsigned_pair<1, 2>,
                                      unsigned_pair<3, 4>,
                                      unsigned_pair<5, 6>>
                                >::value == 1? 1 : -1];
  int check9[X4<1, 3>::Inner<tuple<unsigned_pair<1, 2>,
                                   unsigned_pair<3, 4>,
                                   unsigned_pair<5, 6>>
                             >::value == 0? 1 : -1];

  template<class> struct add_reference;
  template<class> struct add_pointer;
  template<class> struct add_const;

  template<template<class> class ...Templates>
  struct X5 {
    template<typename> struct Inner {
      static const unsigned value = 0;
    };

    template<typename ...Types>
    struct Inner<tuple<Templates<Types>...>> {
      static const unsigned value = 1;
    };
  };

  int check10[X5<add_reference, add_pointer, add_const>
                ::Inner<tuple<add_reference<int>,
                              add_pointer<float>,
                              add_const<double>>>::value == 1? 1 : -1];
  int check11[X5<add_reference, add_pointer>
                ::Inner<tuple<add_reference<int>,
                              add_pointer<float>,
                              add_const<double>>>::value == 0? 1 : -1];

}
