// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -emit-llvm -g %s -o - | FileCheck %s

int foo(int x) {
  if(x)
  {
    class X {
    public:
      char z;
      X(int y) : z(y) {}
    };
    {
      X a(x);
      return a.z;
    }
  }
  return 0;
}

// CHECK: !{{[0-9]+}} = !DICompositeType(tag: DW_TAG_class_type, name: "X", scope: [[LBScope:![0-9]+]],
// CHECK: [[LBScope]] = distinct !DILexicalBlock(scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 5)

