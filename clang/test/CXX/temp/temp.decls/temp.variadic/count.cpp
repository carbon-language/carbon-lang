// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

namespace Basic {
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

namespace WithPackExpansion {
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
