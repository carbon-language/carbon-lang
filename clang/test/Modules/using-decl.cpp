// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -verify

@import using_decl.a;

// expected-no-diagnostics
UsingDecl::using_decl_type x = UsingDecl::using_decl_var;
UsingDecl::inner y = x;
