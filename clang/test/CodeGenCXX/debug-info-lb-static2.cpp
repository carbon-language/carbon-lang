// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -emit-llvm -g %s -o - | FileCheck %s

int main() {
  static int X = 10;
  {
    static bool X = false;
    return (int) X;
  }
  return X;
}

// CHECK: [[FuncScope:![0-9]+]] = !DISubprogram(name: "main",
// CHECK: !{{[0-9]+}} = !DIGlobalVariable(name: "X", scope: [[FuncScope:![0-9]+]],
// CHECK: !{{[0-9]+}} = !DIGlobalVariable(name: "X", scope: [[LBScope:![0-9]+]],
// CHECK: [[LBScope]] = distinct !DILexicalBlock(scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 5)

