// RUN: clang -fsyntax-only -verify %s

template<int I, int J>
struct Bitfields {
  int simple : I; // expected-error{{bit-field 'simple' has zero width}}
  int parens : (J);
};

void test_Bitfields(Bitfields<0, 5> *b) {
  (void)sizeof(Bitfields<10, 5>);
  (void)sizeof(Bitfields<0, 1>); // expected-note{{in instantiation of template class 'struct Bitfields<0, 1>' requested here}}
}
