// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -I%S/Inputs/anon-namespace -verify %s
// expected-no-diagnostics
#include "b1.h"
#include "c.h"
using namespace N;
