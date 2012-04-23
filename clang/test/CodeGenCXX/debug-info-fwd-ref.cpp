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

// Make sure we have two DW_TAG_class_types for baz and bar and no forward
// references.
// FIXME: These should be struct types to match the declaration.
// CHECK: metadata !{i32 {{.*}}, null, metadata !"bar", metadata !6, i32 8, i64 128, i64 64, i32 0, i32 0, null, metadata !18, i32 0, null, null} ; [ DW_TAG_class_type ]
// CHECK: metadata !{i32 {{.*}}, null, metadata !"baz", metadata !6, i32 3, i64 32, i64 32, i32 0, i32 0, null, metadata !21, i32 0, null, null} ; [ DW_TAG_class_type ]
// CHECK-NOT: metadata !{i32 {{.*}}, null, metadata !"bar", metadata !6, i32 8, i64 0, i64 0, i32 0, i32 4, i32 0, null, i32 0, i32 0} ; [ DW_TAG_class_type ]
// CHECK-NOT: metadata !{i32 {{.*}}, null, metadata !"baz", metadata !6, i32 3, i64 0, i64 0, i32 0, i32 4, null, null, i32 0, null, null} ; [ DW_TAG_class_type ]

