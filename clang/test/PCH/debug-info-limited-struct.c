// RUN: %clang_cc1 -emit-pch -o %t %S/debug-info-limited-struct.h
// RUN: %clang_cc1 -include-pch %t -emit-llvm %s -g -o - | FileCheck %s

// CHECK-DAG: [ DW_TAG_structure_type ] [foo] {{.*}} [def]
