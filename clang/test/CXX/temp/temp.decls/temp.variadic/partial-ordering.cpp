// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

// Various tests related to partial ordering of variadic templates.
template<typename ...Types> struct tuple;

template<typename Tuple> 
struct X1 {
  static const unsigned value = 0;
};

template<typename Head, typename ...Tail>
struct X1<tuple<Head, Tail...> > {
  static const unsigned value = 1;
};

template<typename Head, typename ...Tail>
struct X1<tuple<Head, Tail&...> > {
  static const unsigned value = 2;
};

template<typename Head, typename ...Tail>
struct X1<tuple<Head&, Tail&...> > {
  static const unsigned value = 3;
};

int check0[X1<tuple<>>::value == 0? 1 : -1];
int check1[X1<tuple<int>>::value == 2? 1 : -1];
int check2[X1<tuple<int, int>>::value == 1? 1 : -1];
int check3[X1<tuple<int, int&>>::value == 2? 1 : -1];
int check4[X1<tuple<int&, int&>>::value == 3? 1 : -1];
