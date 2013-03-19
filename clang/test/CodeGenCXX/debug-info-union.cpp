// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin -std=c++11 %s -o - | FileCheck %s

union E {
  int a;
  float b;
  int bb() { return a;}
  float aa() { return b;}
  E() { a = 0; }
};

E e;

// CHECK: {{.*}} ; [ DW_TAG_union_type ] [E] [line 3, size 32, align 32, offset 0]
// CHECK: {{.*}} ; [ DW_TAG_subprogram ] [line 6] [bb]
// CHECK: {{.*}} ; [ DW_TAG_subprogram ] [line 7] [aa]
// CHECK: {{.*}} ; [ DW_TAG_subprogram ] [line 8] [E]
