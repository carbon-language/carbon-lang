// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodule-cache-path %t -F %S/Inputs -I %S/Inputs -verify %s

#import "point.h"
__import_module__ Module;
#import "point.h"

