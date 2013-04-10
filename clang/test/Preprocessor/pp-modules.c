// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -x objective-c %s -F %S/../Modules/Inputs -E -o - | FileCheck %s

// CHECK: int bar();
int bar();
// CHECK: @import Module; /* clang -E: implicit import for "{{.*Headers[/\\]Module.h}}" */
#include <Module/Module.h>
// CHECK: int foo();
int foo();
// CHECK: @import Module; /* clang -E: implicit import for "{{.*Headers[/\\]Module.h}}" */
#include <Module/Module.h>
