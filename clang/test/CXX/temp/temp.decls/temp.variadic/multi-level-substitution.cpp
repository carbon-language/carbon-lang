// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

template<typename T, T ...Values> struct value_tuple {};
template<typename...> struct tuple { };
template<typename T, typename U> struct pair { };

template<typename T, T Value> struct value_c;

template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

template<typename T>
struct X0 {
  template<T ...Values>
  void f(value_tuple<T, Values...> * = 0);
};

void test_X0() {
  X0<int>().f<1, 2, 3, 4, 5>();
}

namespace PacksAtDifferentLevels {

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

namespace ExpandingNonTypeTemplateParameters {
  template<typename ...Types>
  struct tuple_of_values {
    template<Types ...Values> // expected-error{{a non-type template parameter cannot have type 'float'}} \
    // expected-note{{template parameter is declared here}}
    struct apply { // expected-note 2{{template is declared here}}
      typedef tuple<value_c<Types, Values>...> type;
    };
  };

  int i;
  float f;
  int check_tuple_of_values_1[
        is_same<tuple_of_values<int&, float&, char, int>::apply<i, f, 'a', 17>
                  ::type,
                tuple<value_c<int&, i>, value_c<float&, f>, value_c<char, 'a'>,
                      value_c<int, 17>>
                >::value? 1 : -1];

  tuple_of_values<int, float> tv1; // expected-note{{in instantiation of template class 'ExpandingNonTypeTemplateParameters::tuple_of_values<int, float>' requested here}}

  tuple_of_values<int&, float&>::apply<i, i>::type tv2; // expected-error{{non-type template parameter of reference type 'float &' cannot bind to template argument of type 'int'}}

  tuple_of_values<int&, float&>::apply<i>::type tv3; // expected-error{{too few template arguments for class template 'apply'}}

  tuple_of_values<int&, float&>::apply<i, f, i>::type tv4; // expected-error{{too many template arguments for class template 'apply'}}
}

namespace ExpandingFunctionParameters {
  template<typename ...T>
  struct X0 {
    typedef int type;
  };

  template<typename ...T>
  struct X1 {
    template<typename ... U>
    typename X0<T(T, U...)...>::type f(U...);
  };

  void test() {
    X1<float> x1;
    x1.f(17, 3.14159);
  }
}

namespace PR10230 {
  template<typename>
  struct s
  {
    template<typename... Args>
    auto f() -> int(&)[sizeof...(Args)];
  };

  void main()
  {
    int (&ir1)[1] = s<int>().f<int>();
    int (&ir3)[3] = s<int>().f<int, float, double>();
  }
}
