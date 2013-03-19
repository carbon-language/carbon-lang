// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s

struct baz {
    int h;
    baz(int a) : h(a) {}
};

struct bar {
    baz b;
    baz& b_ref;
    bar(int x) : b(x), b_ref(b) {}
};

int main(int argc, char** argv) {
    bar myBar(1);
    return 0;
}

// Make sure we have two DW_TAG_structure_types for baz and bar and no forward
// references.
// CHECK-NOT: [fwd]
// CHECK: [ DW_TAG_structure_type ] [bar]
// CHECK: [ DW_TAG_structure_type ] [baz]
// CHECK-NOT: [fwd]
