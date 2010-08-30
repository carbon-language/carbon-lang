// RUN: %clang_cc1 %s -fsyntax-only -verify -fms-extensions


// ::type_info is predeclared with forward class declartion
void f(const type_info &a);


