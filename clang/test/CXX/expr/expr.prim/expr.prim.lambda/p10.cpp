// RUN: %clang_cc1 -std=c++11 %s -verify

int GlobalVar; // expected-note {{declared here}}

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
    int variable; // expected-note {{declared here}}
    (void)[&Overload] () {}; // expected-error {{does not name a variable}} 
    (void)[&GlobalVar] () {}; // expected-error {{does not have automatic storage duration}} 
    (void)[&AmbiguousVar] () {}; // expected-error {{reference to 'AmbiguousVar' is ambiguous}} 
    (void)[&Variable] () {}; // expected-error {{use of undeclared identifier 'Variable'; did you mean 'variable'}}
  }
};
