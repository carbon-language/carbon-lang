// RUN: %clang_cc1 -emit-llvm -g %s -o - | FileCheck %s
typedef int v4si __attribute__((__vector_size__(16)));

v4si a;

// Test that we get an array type that's also a vector out of debug.
// CHECK: [ DW_TAG_array_type ] [line 0, size 128, align 128, offset 0] [vector] [from int]
