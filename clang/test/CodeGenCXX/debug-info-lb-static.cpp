// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -emit-llvm -g %s -o - | FileCheck %s

int foo(int x) {
  if(x)
  {
    static int bar = 0;
    {
      int a = bar++;
      return a;
    }
  }
  return 0;
}

// CHECK: !{{[0-9]+}} = !DIGlobalVariable(name: "bar", scope: [[LBScope:![0-9]+]],
// CHECK: [[LBScope]] = distinct !DILexicalBlock(scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 5)

