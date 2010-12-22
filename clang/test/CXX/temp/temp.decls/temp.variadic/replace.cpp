// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

// Simple metafunction that replaces the template arguments of
// template template parameters with 'int'.
template<typename T>
struct EverythingToInt;

template<template<typename ...> class TT, typename T1, typename T2>
struct EverythingToInt<TT<T1, T2> > {
  typedef TT<int, int> type;
};

template<typename...> struct tuple { };

int check0[is_same<EverythingToInt<tuple<double, float>>::type, 
                   tuple<int, int>>::value? 1 : -1];
