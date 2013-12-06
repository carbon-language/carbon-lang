// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules-cache-path=%t -fmodules -fmodules-decluse -fmodule-name=XH -I %S/Inputs/declare-use %s -verify

#include "h.h"
#include "e.h"
#include "f.h" // expected-error {{does not depend on a module exporting}}
const int h2 = h1+e+f;
