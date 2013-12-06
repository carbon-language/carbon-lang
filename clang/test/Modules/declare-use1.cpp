// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules-cache-path=%t -fmodules -fmodules-decluse -fmodule-name=XG -I %S/Inputs/declare-use %s -verify

#include "g.h"
#include "e.h"
#include "f.h" // expected-error {{module XG does not depend on a module exporting 'f.h'}}
const int g2 = g1+e+f;
