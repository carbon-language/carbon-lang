// RUN: %clang_cc1 %s -triple x86_64-apple-darwin -emit-llvm -o - | FileCheck %s
// Exercise various use cases for local asm "register variables".

int foo() {
// CHECK: %a = alloca i32

  register int a asm("rsi")=5;
// CHECK: store i32 5, i32* %a

  asm volatile("; %0 This asm defines rsi" : "=r"(a));
// CHECK: %0 = call i32 asm sideeffect "; $0 This asm defines rsi", "={rsi},~{dirflag},~{fpsr},~{flags}"()
// CHECK: store i32 %0, i32* %a

  a = 42;
// CHECK:  store i32 42, i32* %a

  asm volatile("; %0 This asm uses rsi" : : "r"(a));
// CHECK:  %tmp = load i32* %a
// CHECK:  call void asm sideeffect "; $0 This asm uses rsi", "{rsi},~{dirflag},~{fpsr},~{flags}"(i32 %tmp)

  return a;
// CHECK:  %tmp1 = load i32* %a
// CHECK:  ret i32 %tmp1
}
