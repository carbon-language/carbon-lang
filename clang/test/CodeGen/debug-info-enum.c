// RUN: %clang_cc1 -emit-llvm -g %s -o - | FileCheck %s

// CHECK: [[TEST3_ENUMS:![0-9]*]], null, null, null} ; [ DW_TAG_enumeration_type ] [e]
// CHECK: [[TEST3_ENUMS]] = !{[[TEST3_E:![0-9]*]]}
// CHECK: [[TEST3_E]] = !{!"0x28\00E\00-1"} ; [ DW_TAG_enumerator ] [E :: -1]

enum e;
void func(enum e *p) {
}
enum e { E = -1 };
