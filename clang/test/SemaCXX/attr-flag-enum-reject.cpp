// RUN: %clang_cc1 -verify -fsyntax-only -x c++ -Wassign-enum %s

enum __attribute__((flag_enum)) flag { // expected-warning {{ignored}}
};
