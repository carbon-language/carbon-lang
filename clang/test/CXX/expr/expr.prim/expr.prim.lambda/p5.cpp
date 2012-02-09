// RUN: %clang_cc1 -std=c++11 %s -verify

int test_default_args() {
  (void)[](int i = 5,  // expected-error{{default arguments can only be specified for parameters in a function declaration}}
     int j = 17) {}; // expected-error{{default arguments can only be specified for parameters in a function declaration}}
}
