// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s

void testVLAwithSize(int s)
{
// CHECK: metadata !{metadata !"0x100\00vla\00[[@LINE+1]]\008192", metadata {{.*}}, metadata {{.*}}, metadata {{.*}}} ; [ DW_TAG_auto_variable ] [vla] [line [[@LINE+1]]]
  int vla[s];
  int i;
  for (i = 0; i < s; i++) {
    vla[i] = i*i;
  }
}
