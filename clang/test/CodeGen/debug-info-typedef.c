// RUN: %clang_cc1 -emit-llvm -g -I%p %s -o - | FileCheck %s
// Test that the location of the typedef points to the header file.
#line 1 "a.c"
#line 2 "b.h"
typedef int MyType;
#line 2 "a.c"

MyType a;

// CHECK:  !"0x16\00MyType\002\00{{.*}}", ![[HEADER:[0-9]+]], null{{.*}}} ; [ DW_TAG_typedef ] [MyType] [line 2, size 0, align 0, offset 0] [from int]
// CHECK: ![[HEADER]] = !{!"b.h",
