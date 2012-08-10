// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -O0 -fms-extensions -fenable-experimental-ms-inline-asm -w -emit-llvm -o - | FileCheck %s

void t1() {
// CHECK: @t1
// CHECK: call void asm sideeffect "", "~{dirflag},~{fpsr},~{flags}"() nounwind ia_nsdialect
// CHECK: ret void
  __asm {}
}
