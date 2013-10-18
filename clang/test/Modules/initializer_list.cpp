// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules -fmodules-cache-path=%t -I %S/Inputs %s -verify -std=c++11

// expected-no-diagnostics
@import initializer_list;

int n = std::min({1, 2, 3});
