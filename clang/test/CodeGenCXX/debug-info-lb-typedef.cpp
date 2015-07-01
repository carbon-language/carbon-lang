// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -emit-llvm -g %s -o - | FileCheck %s

int foo(int x) {
  if(x) 
  {
    typedef char X;
    {
      X a = x;
      return a;
    }
  }
  return 0;
}

// CHECK: !{{[0-9]+}} = !DIDerivedType(tag: DW_TAG_typedef, name: "X", scope: [[LBScope:![0-9]+]],
// CHECK: [[LBScope]] = distinct !DILexicalBlock(scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 5)

