// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x objective-c %s -F %S/../Modules/Inputs -E -frewrite-includes -o - | FileCheck %s

int bar();
#include <Module/Module.h>
int foo();
#include <Module/Module.h>

// CHECK: int bar();{{$}}
// CHECK-NEXT: #if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: #include <Module/Module.h>{{$}}
// CHECK-NEXT: #endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: # 5 "{{.*[/\\]}}rewrite-includes-modules.c"{{$}}
// CHECK-NEXT: @import Module; /* clang -frewrite-includes: implicit import */{{$}}
// CHECK-NEXT: # 6 "{{.*[/\\]}}rewrite-includes-modules.c"{{$}}
// CHECK-NEXT: int foo();{{$}}
// CHECK-NEXT: #if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: #include <Module/Module.h>{{$}}
// CHECK-NEXT: #endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: # 7 "{{.*[/\\]}}rewrite-includes-modules.c"{{$}}
// CHECK-NEXT: @import Module; /* clang -frewrite-includes: implicit import */{{$}}
// CHECK-NEXT: # 8 "{{.*[/\\]}}rewrite-includes-modules.c"{{$}}
