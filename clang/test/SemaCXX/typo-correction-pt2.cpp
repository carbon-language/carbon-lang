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
