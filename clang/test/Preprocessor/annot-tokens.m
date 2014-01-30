// RUN: %clang_cc1 -E %s -fmodules -fmodules-cache-path=%t -I%S/Inputs | FileCheck %s
// CHECK: @import a; /* clang -E: implicit import
#include "a.h"
// CHECK: #pragma clang __debug parser_crash
#pragma clang __debug parser_crash
