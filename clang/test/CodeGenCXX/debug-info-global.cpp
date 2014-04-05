// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -emit-llvm -g %s -o - | FileCheck %s

// Multiple references to the same constant should result in only one entry in
// the globals list.

const int cnst = 42;
int f1() {
  return cnst + cnst;
}

// CHECK: metadata [[GLOBALS:![0-9]*]], metadata {{![0-9]*}}, metadata !"{{.*}}", i32 {{[0-9]*}}} ; [ DW_TAG_compile_unit ]

// CHECK: [[GLOBALS]] = metadata !{metadata [[CNST:![0-9]*]]}

