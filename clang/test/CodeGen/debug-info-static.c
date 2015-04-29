// RUN: %clang_cc1  -g -emit-llvm -o - %s | FileCheck %s

// CHECK: !DIGlobalVariable({{.*}}variable: i32* @f.xyzzy
void f(void)
{
   static int xyzzy;
   xyzzy += 3;
}
