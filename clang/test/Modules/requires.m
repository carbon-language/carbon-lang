// RUN: rm -rf %t
// RUN: %clang_cc1 -Wauto-import -fmodule-cache-path %t -fmodules -F %S/Inputs %s -verify

__import_module__ DependsOnModule.CXX; // expected-error{{module 'DependsOnModule.CXX' requires feature 'cplusplus'}}

