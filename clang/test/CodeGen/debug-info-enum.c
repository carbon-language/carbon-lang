// RUN: %clang_cc1 -emit-llvm -g %s -o - | FileCheck %s

// CHECK: !MDCompositeType(tag: DW_TAG_enumeration_type, name: "e"
// CHECK-SAME:             elements: [[TEST3_ENUMS:![0-9]*]]
// CHECK: [[TEST3_ENUMS]] = !{[[TEST3_E:![0-9]*]]}
// CHECK: [[TEST3_E]] = !MDEnumerator(name: "E", value: -1)

enum e;
void func(enum e *p) {
}
enum e { E = -1 };
