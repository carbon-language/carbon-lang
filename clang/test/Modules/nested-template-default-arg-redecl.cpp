// RUN: rm -rf %t
// RUN: %clang_cc1 -x c++ -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:     -I %S/Inputs/nested-template-default-arg-redecl -std=c++14 \
// RUN:     -fmodules-local-submodule-visibility -verify %s
#include "alias2.h"
#include "var2.h"
#include "strct2.h"
#include "func2.h"

// FIXME: Variable templates lexical decl context appears to be the translation
// unit, which is incorrect. Fixing this will hopefully address the following
// error/bug:

// expected-note@Inputs/nested-template-default-arg-redecl/var.h:4 {{default argument declared here}}
auto var = &var_outer::var<>; // expected-error {{default argument of 'var' must be imported from module 'VAR1' before it is required}}
auto func = &func_outer::func<>;
strct_outer::strct<> *strct;
alias_outer::alias<> *alias;
