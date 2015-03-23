// RUN: rm -rf %t
// RUN: %clang_cc1 -fexceptions -fcxx-exceptions -fmodules -fmodules-cache-path=%t -I%S/Inputs/update-exception-spec -emit-llvm-only %s
#include "a.h"
void use(B *p);
#include "c.h"
void use(B *p) { g(p); }
