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

// CHECK: {{.*}} [ DW_TAG_structure_type ] [I]
// Check for "a".
// CHECK: {{.*}} [ DW_TAG_member ] [a] [line 7, size 32, align 32, offset 0] [public] [from int]
// Make sure we don't output the same type twice.
// CHECK-NOT: {{.*}} [ DW_TAG_structure_type ] [I]
// Check for "b".
// CHECK: {{.*}} [ DW_TAG_member ] [b] [line 18, size 32, align 32, offset 0] [public] [from int]
