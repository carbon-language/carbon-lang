// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

template<typename ...Types> struct tuple;

namespace PackExpansionNotAtEnd {
  template<typename T, typename U>
  struct tuple_same_with_int {
    static const bool value = false;
  };

  template<typename ...Types>
  struct tuple_same_with_int<tuple<Types...>, tuple<Types..., int>> {
    static const bool value = true;
  };

  int tuple_same_with_int_1[tuple_same_with_int<tuple<int, float, double>,
                                                tuple<int, float, double, int>
                                                >::value? 1 : -1];

  template<typename ... Types> struct UselessPartialSpec;

           template<typename ... Types, // expected-note{{non-deducible template parameter 'Types'}}
           typename Tail> // expected-note{{non-deducible template parameter 'Tail'}}
  struct UselessPartialSpec<Types..., Tail>; // expected-warning{{class template partial specialization contains template parameters that can not be deduced; this partial specialization will never be used}}
}
