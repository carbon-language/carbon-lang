// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fmodules-decluse -I %S/Inputs/string_names %s -fmodule-name="my/module-a" -verify

#include "a.h"
#include "b.h" // expected-error {{does not depend on a module exporting}}
#include "c.h"
