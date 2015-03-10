// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -I%S/Inputs/PR21687 -emit-llvm-only %s
#include "c.h"
