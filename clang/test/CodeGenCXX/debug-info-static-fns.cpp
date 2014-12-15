// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s

namespace A {
  static int a(int b) { return b + 4; }

  int b(int c) { return c + a(c); }
}

// Verify that a is present and mangled.
// CHECK: !"0x2e\00a\00a\00_ZN1AL1aEi\00{{.*}}", {{.*}}, i32 (i32)* @_ZN1AL1aEi, {{.*}} ; [ DW_TAG_subprogram ] [line 4] [local] [def] [a]
