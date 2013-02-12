// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules -fmodules-cache-path=%t -I %S/Inputs %s -verify -std=c++11

@import cxx_inline_namespace;

T x; // expected-error {{unknown type name 'T'}}
