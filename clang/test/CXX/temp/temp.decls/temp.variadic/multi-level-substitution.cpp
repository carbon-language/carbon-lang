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
}
