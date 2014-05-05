// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -g %s -o - | FileCheck %s

// Make sure we generate debug symbols for an indirectly referenced
// extension to an interface.

// This happens to be the order the members are emitted in... I'm assuming it's
// not meaningful/important, so if something causes the order to change, feel
// free to update the test to reflect the new order.
// CHECK: ; [ DW_TAG_member ] [a]
// CHECK: ; [ DW_TAG_member ] [d]
// CHECK: ; [ DW_TAG_member ] [c]
// CHECK: ; [ DW_TAG_member ] [b]

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


I *source();

@interface I()
{
    @public int c;
}
@end

void use() {
    int _c = source()->c;
}

@interface I()
{
    @public int d;
}
@end

I *x();
