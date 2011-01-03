// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

// This is a collection of various template metafunctions involving
// variadic templates, which are meant to exercise common use cases.
template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

template<typename...> struct tuple { };
template<int ...> struct int_tuple { };

namespace Count {
  template<typename Head, typename ...Tail>
  struct count {
    static const unsigned value = 1 + count<Tail...>::value;
  };

  template<typename T>
  struct count<T> {
    static const unsigned value = 1;
  };

  int check1[count<int>::value == 1? 1 : -1];
  int check2[count<float, double>::value == 2? 1 : -1];
  int check3[count<char, signed char, unsigned char>::value == 3? 1 : -1];
}

namespace CountWithPackExpansion {
  template<typename ...> struct count;

  template<typename Head, typename ...Tail>
  struct count<Head, Tail...> {
    static const unsigned value = 1 + count<Tail...>::value;
  };

  template<>
  struct count<> {
    static const unsigned value = 0;
  };

  int check0[count<>::value == 0? 1 : -1];
  int check1[count<int>::value == 1? 1 : -1];
  int check2[count<float, double>::value == 2? 1 : -1];
  int check3[count<char, signed char, unsigned char>::value == 3? 1 : -1];
}

namespace Replace {
  // Simple metafunction that replaces the template arguments of
  // template template parameters with 'int'.
  template<typename T>
  struct EverythingToInt;

  template<template<typename ...> class TT, typename T1, typename T2>
  struct EverythingToInt<TT<T1, T2> > {
    typedef TT<int, int> type;
  };

  int check0[is_same<EverythingToInt<tuple<double, float>>::type, 
             tuple<int, int>>::value? 1 : -1];
}
