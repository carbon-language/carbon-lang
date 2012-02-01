// RUN: %clang_cc1 -std=c++11 -fblocks %s -verify

void block_capture_errors() {
  __block int var; // expected-note 2{{'var' declared here}}
  (void)[var] { }; // expected-error{{__block variable 'var' cannot be captured in a lambda}} \
  // expected-error{{lambda expressions are not supported yet}}

  // FIXME: this should produce the same error as above
  (void)[=] { var = 17; }; // expected-error{{reference to local variable 'var' declared in enclosed function 'block_capture_errors'}} \
  // expected-error{{lambda expressions are not supported yet}}
}
