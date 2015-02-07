// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -I%S/Inputs/deferred-lookup -verify %s
// expected-no-diagnostics

namespace N { int f(int); }
#include "b.h"
