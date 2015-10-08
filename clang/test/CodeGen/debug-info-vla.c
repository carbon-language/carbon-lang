// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple x86_64-apple-darwin %s -o - | FileCheck %s

void testVLAwithSize(int s)
{
// CHECK: dbg.declare
// CHECK: dbg.declare({{.*}}, metadata ![[VAR:.*]], metadata ![[EXPR:.*]])
// CHECK: ![[VAR]] = !DILocalVariable(name: "vla",{{.*}} line: [[@LINE+2]]
// CHECK: ![[EXPR]] = !DIExpression(DW_OP_deref)
  int vla[s];
  int i;
  for (i = 0; i < s; i++) {
    vla[i] = i*i;
  }
}
