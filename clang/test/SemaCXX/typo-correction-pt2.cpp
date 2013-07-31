// RUN: %clang_cc1 -fsyntax-only -verify -Wno-c++11-extensions %s
//
// FIXME: This file is overflow from test/SemaCXX/typo-correction.cpp due to a
// hard-coded limit of 20 different typo corrections Sema::CorrectTypo will
// attempt within a single file (which is to avoid having very broken files take
// minutes to finally be rejected by the parser).

namespace PR12287 {
class zif {
  void nab(int);
};
void nab();  // expected-note{{'::PR12287::nab' declared here}}
void zif::nab(int) {
  nab();  // expected-error{{too few arguments to function call, expected 1, have 0; did you mean '::PR12287::nab'?}}
}
}

namespace TemplateFunction {
template <class T>  // expected-note {{'::TemplateFunction::A' declared here}}
void A(T) { }

template <class T>  // expected-note {{'::TemplateFunction::B' declared here}}
void B(T) { }

class Foo {
 public:
  void A(int, int) {}
  void B() {}
};

void test(Foo F, int num) {
  F.A(num);  // expected-error {{too few arguments to function call, expected 2, have 1; did you mean '::TemplateFunction::A'?}}
  F.B(num);  // expected-error {{too many arguments to function call, expected 0, have 1; did you mean '::TemplateFunction::B'?}}
}
}
