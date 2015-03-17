// RUN: %clang_cc1 -std=c++11 -fsyntax-only -Wdocumentation-unknown-command -verify %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -Werror -Wno-documentation-unknown-command %s

// expected-warning@+1 {{unknown command tag name}}
/// aaa \unknown
int test_unknown_comand_1;

// expected-warning@+1 {{unknown command tag name 'retur'; did you mean 'return'?}}
/// \retur aaa
int test_unknown_comand_2();

