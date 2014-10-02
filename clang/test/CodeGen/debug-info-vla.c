// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s

void testVLAwithSize(int s)
{
// CHECK: metadata !{i32 {{.*}}, metadata {{.*}}, metadata !"vla", metadata {{.*}}, i32 [[@LINE+1]], metadata {{.*}}, i32 8192, i32 0} ; [ DW_TAG_auto_variable ] [vla] [line [[@LINE+1]]]
  int vla[s];
  int i;
  for (i = 0; i < s; i++) {
    vla[i] = i*i;
  }
}
