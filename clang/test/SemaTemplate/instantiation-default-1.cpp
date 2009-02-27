// RUN: clang -fsyntax-only -verify %s

template<typename T, typename U = const T> struct Def1;

template<> struct Def1<int> { 
  void foo();
};

template<> struct Def1<const int> { // expected-note{{previous definition is here}}
  void bar();
};

void test_Def1(Def1<int, const int> *d1, Def1<const int, const int> *d2) {
  d1->foo();
  d2->bar();
}

template<> struct Def1<const int, const int> { }; // expected-error{{redefinition of 'Def1'}}

