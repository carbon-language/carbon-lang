// RUN: rm -rf %t
// RUN: %clang_cc1 -Wauto-import -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs %s -verify

@import DependsOnModule.CXX;
@import DependsOnModule.NotCXX; // expected-error{{module 'DependsOnModule.NotCXX' is incompatible with feature 'cplusplus'}}
@import DependsOnModule.NotObjC; // expected-error{{module 'DependsOnModule.NotObjC' is incompatible with feature 'objc'}}
