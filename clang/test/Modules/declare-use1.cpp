// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules-cache-path=%t -fmodules -fmodules-decluse -fmodule-name=XG -I %S/Inputs/declare-use %s -verify

#include "g.h"
#include "e.h"
#include "f.h" // expected-error {{use of a module not declared used}}
const int g2 = g1+e+f;
