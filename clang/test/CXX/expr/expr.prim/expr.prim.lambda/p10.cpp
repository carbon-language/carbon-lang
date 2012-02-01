// RUN: %clang_cc1 -std=c++11 %s -verify

int GlobalVar; // expected-note 2{{declared here}}

namespace N {
  int AmbiguousVar; // expected-note {{candidate}}
}
int AmbiguousVar; // expected-note {{candidate}}
using namespace N;

class X0 {
  int Member;

  static void Overload(int);
  void Overload();
  virtual X0& Overload(float);

  void explicit_capture() {
    [&Overload] () {}; // expected-error {{does not name a variable}} expected-error {{not supported yet}}
    [&GlobalVar] () {}; // expected-error {{does not have automatic storage duration}} expected-error {{not supported yet}}
    [&AmbiguousVar] () {} // expected-error {{reference to 'AmbiguousVar' is ambiguous}} expected-error {{not supported yet}}
    [&Globalvar] () {}; // expected-error {{use of undeclared identifier 'Globalvar'; did you mean 'GlobalVar}}
  }
};
