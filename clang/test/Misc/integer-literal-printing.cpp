// RUN: %clang_cc1 %s -fsyntax-only -verify

// PR11179
template <short T> class Type1 {};
template <short T> void Function1(Type1<T>& x) {} // expected-note{{candidate function [with T = -42] not viable: no known conversion from 'Type1<-42>' to 'Type1<-42> &' for 1st argument;}}

template <unsigned short T> class Type2 {};
template <unsigned short T> void Function2(Type2<T>& x) {} // expected-note{{candidate function [with T = 42] not viable: no known conversion from 'Type2<42>' to 'Type2<42> &' for 1st argument;}}

template <__int128_t T> class Type3 {};
template <__int128_t T> void Function3(Type3<T>& x) {} // expected-note{{candidate function [with T = -42] not viable: no known conversion from 'Type3<-42>' to 'Type3<-42i128> &' for 1st argument;}}

template <__uint128_t T> class Type4 {};
template <__uint128_t T> void Function4(Type4<T>& x) {} // expected-note{{candidate function [with T = 42] not viable: no known conversion from 'Type4<42>' to 'Type4<42Ui128> &' for 1st argument;}}

void Function() {
  Function1(Type1<-42>()); // expected-error{{no matching function for call to 'Function1'}}
  Function2(Type2<42>()); // expected-error{{no matching function for call to 'Function2'}}
  Function3(Type3<-42>()); // expected-error{{no matching function for call to 'Function3'}}
  Function4(Type4<42>()); // expected-error{{no matching function for call to 'Function4'}}
}
