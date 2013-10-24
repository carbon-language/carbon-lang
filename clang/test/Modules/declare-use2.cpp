// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules-cache-path=%t -fmodules -fmodules-decluse -fmodule-name=XH -I %S/Inputs/declare-use %s -verify

#include "h.h"
#include "e.h"
#include "f.h" // expected-error {{use of a module not declared used}}
const int h2 = h1+e+f;
