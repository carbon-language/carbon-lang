// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin -std=c++11 %s -o - | FileCheck %s

union E {
  int a;
  float b;
  int bb() { return a;}
  float aa() { return b;}
  E() { a = 0; }
};

E e;

// CHECK: metadata !{i32 {{.*}}, null, metadata !"E", metadata !6, i32 3, i64 32, i64 32, i64 0, i32 0, null, metadata !11, i32 0, null} ; [ DW_TAG_union_type ]
// CHECK: metadata !{i32 {{.*}}, i32 0, metadata !10, metadata !"bb", metadata !"bb", metadata !"_ZN1E2bbEv", metadata !6, i32 6, metadata !17, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !19, i32 6} ; [ DW_TAG_subprogram ]
// CHECK: metadata !{i32 {{.*}}, i32 0, metadata !10, metadata !"aa", metadata !"aa", metadata !"_ZN1E2aaEv", metadata !6, i32 7, metadata !22, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !24, i32 7} ; [ DW_TAG_subprogram ]
// CHECK: metadata !{i32 {{.*}}, i32 0, metadata !10, metadata !"E", metadata !"E", metadata !"", metadata !6, i32 8, metadata !7, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !27, i32 8} ; [ DW_TAG_subprogram ]
