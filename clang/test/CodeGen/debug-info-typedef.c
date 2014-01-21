// RUN: %clang_cc1 -emit-llvm -g -I%p %s -o - | FileCheck %s
// Test that the location of the typedef points to the header file.
#include "typedef.h"

MyType a;

// CHECK: metadata ![[HEADER:[0-9]+]]} ; [ DW_TAG_typedef ] [MyType] [line 1, size 0, align 0, offset 0] [from int]
// CHECK: ![[HEADER]] = {{.*}}debug-info-typedef.h",
