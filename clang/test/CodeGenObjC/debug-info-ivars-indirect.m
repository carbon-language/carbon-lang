// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -g %s -o - | FileCheck %s

// Make sure we generate debug symbols for an indirectly referenced
// extension to an interface.

@interface I
{
    @public int a;
}
@end

void foo(I* pi) {
    int _a = pi->a;
}

// another layer of indirection
struct S
{
    I* i;
};

@interface I()
{
    @public int b;
}
@end

void gorf (struct S* s) {
    int _b = s->i->b;
}

// CHECK: metadata !{i32 {{[0-9]*}}, metadata !{{[0-9]*}}, metadata !"b", metadata !{{[0-9]*}}, i32 24, i64 32, i64 32, i64 0, i32 0, metadata !{{[0-9]*}}, null} ; [ DW_TAG_member ] [b] [line 24, size 32, align 32, offset 0] [from int]
