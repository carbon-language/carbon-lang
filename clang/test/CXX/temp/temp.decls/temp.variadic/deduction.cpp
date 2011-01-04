// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

namespace DeductionForInstantiation {
  template<unsigned I, typename ...Types>
  struct X { };

  template<typename ...Types>
  void f0(X<sizeof...(Types), Types&...>) { }

  // No explicitly-specified arguments
  template void f0(X<0>);
  template void f0(X<1, int&>);
  template void f0(X<2, int&, short&>);

  // One explicitly-specified argument
  template void f0<float>(X<1, float&>);
  template void f0<double>(X<1, double&>);

  // Two explicitly-specialized arguments
  template void f0<char, unsigned char>(X<2, char&, unsigned char&>);
  template void f0<signed char, char>(X<2, signed char&, char&>);

  // FIXME: Extension of explicitly-specified arguments
  //  template void f0<short, int>(X<3, short&, int&, long&>);
}
