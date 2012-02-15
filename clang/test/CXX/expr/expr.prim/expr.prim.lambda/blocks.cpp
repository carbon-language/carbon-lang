// RUN: %clang_cc1 -std=c++11 -fblocks %s -verify

void block_capture_errors() {
  __block int var; // expected-note 2{{'var' declared here}}
  (void)[var] { }; // expected-error{{__block variable 'var' cannot be captured in a lambda}}

  (void)[=] { var = 17; }; // expected-error{{__block variable 'var' cannot be captured in a lambda}}
}

void conversion_to_block(int captured) {
  int (^b1)(int) = [=](int x) { return x + captured; };

  const auto lambda = [=](int x) { return x + captured; };
  int (^b2)(int) = lambda;
}
