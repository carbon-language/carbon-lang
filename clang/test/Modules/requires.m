// RUN: rm -rf %t
// RUN: %clang_cc1 -Wauto-import -fmodules-cache-path=%t -fmodules -F %S/Inputs %s -verify

@import DependsOnModule.CXX; // expected-error{{module 'DependsOnModule.CXX' requires feature 'cplusplus'}}

