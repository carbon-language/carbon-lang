// RUN: %clang_cc1 -emit-llvm -g -I%p %s -o - | FileCheck %s
// Test that the location of the typedef points to the header file.
#line 1 "a.c"
#line 2 "b.h"
typedef int MyType;
#line 2 "a.c"

MyType a;

// CHECK: metadata ![[HEADER:[0-9]+]], null, metadata !"MyType"{{.*}} ; [ DW_TAG_typedef ] [MyType] [line 2, size 0, align 0, offset 0] [from int]
// CHECK: ![[HEADER]] = metadata !{metadata !"b.h",
