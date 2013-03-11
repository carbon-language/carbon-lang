// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -g %s -o - | FileCheck %s

// Make sure we generate debug symbols for ivars added by a class extension.

@interface I
{
    @public int a;
}
@end

void foo(I* pi) {
    // poking into pi for primary class ivars.
    int _a = pi->a;
}

@interface I()
{
    @public int b;
}
@end

void gorf (I* pg) {
    // poking into pg for ivars for class extension
    int _b = pg->b;
}

// CHECK: metadata !{i32 {{[0-9]*}}, metadata !{{[0-9]*}}, metadata !"I", {{.*}}} ; [ DW_TAG_structure_type ]
// Check for "a".
// CHECK: metadata !{i32 {{[0-9]*}}, metadata !{{[0-9]*}}, metadata !"a", metadata !{{[0-9]*}}, i32 7, i64 32, i64 32, i64 0, i32 0, metadata !{{[0-9]*}}, null} ; [ DW_TAG_member ] [a] [line 7, size 32, align 32, offset 0] [from int]
// Make sure we don't output the same type twice.
// CHECK-NOT: metadata !{i32 {{[0-9]*}}, metadata !{{[0-9]*}}, metadata !"I", {{.*}}} ; [ DW_TAG_structure_type ]
// Check for "b".
// CHECK: metadata !{i32 {{[0-9]*}}, metadata !{{[0-9]*}}, metadata !"b", metadata !{{[0-9]*}}, i32 18, i64 32, i64 32, i64 0, i32 0, metadata !{{[0-9]*}}, null} ; [ DW_TAG_member ] [b] [line 18, size 32, align 32, offset 0] [from int]
