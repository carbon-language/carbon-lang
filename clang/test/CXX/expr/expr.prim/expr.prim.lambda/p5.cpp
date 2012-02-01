// RUN: %clang_cc1 -std=c++11 %s -verify

int test_default_args() {
  [](int i = 5,  // expected-error{{default arguments can only be specified for parameters in a function declaration}} \
                 // expected-error{{lambda expressions are not supported yet}}
     int j = 17) {}; // expected-error{{default arguments can only be specified for parameters in a function declaration}}
}
