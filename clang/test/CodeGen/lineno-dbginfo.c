// RUN: echo "#include <stddef.h>" > %t.h
// RUN: %clang_cc1 -S -g -include %t.h %s -emit-llvm -o - | FileCheck %s

// CHECK: !MDGlobalVariable(name: "outer",
// CHECK-NOT:               linkageName:
// CHECK-SAME:              line: [[@LINE+2]]
// CHECK-SAME:              isDefinition: true
int outer = 42;

