// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s

namespace A {
  static int a(int b) { return b + 4; }

  int b(int c) { return c + a(c); }
}

// Verify that a is present and mangled.
// CHECK: metadata !{i32 {{.*}}, i32 0, metadata !{{.*}}, metadata !"a", metadata !"a", metadata !"_ZN1AL1aEi", metadata !{{.*}}, i32 4, metadata !{{.*}}, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @_ZN1AL1aEi, null, null, metadata !{{.*}}, i32 4} ; [ DW_TAG_subprogram ]
