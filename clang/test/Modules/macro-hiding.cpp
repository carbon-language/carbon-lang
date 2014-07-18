// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s
#include "c1.h"
#include "b2.h"

void h() { assert(true); }
