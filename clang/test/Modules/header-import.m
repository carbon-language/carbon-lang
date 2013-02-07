// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -F %S/Inputs -I %S/Inputs -verify %s
// expected-no-diagnostics

#import "point.h"
@import Module;
#import "point.h"

