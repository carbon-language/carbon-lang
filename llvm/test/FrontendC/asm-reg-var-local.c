// RUN: %llvmgcc %s -S -o - | FileCheck %s
// Exercise various use cases for local asm "register variables".
// XFAIL: *
// XTARGET: x86_64,i686,i386

int foo() {
// CHECK: %a = alloca i32

  register int a asm("rsi")=5;
// CHECK: store i32 5, i32* %a, align 4

  asm volatile("; %0 This asm defines rsi" : "=r"(a));
// CHECK: %1 = call i32 asm sideeffect "; $0 This asm defines rsi", "={rsi}
// CHECK: store i32 %1, i32* %a

  a = 42;
// CHECK:  store i32 42, i32* %a, align 4

  asm volatile("; %0 This asm uses rsi" : : "r"(a));
// CHECK:  %2 = load i32* %a, align 4
// CHECK:  call void asm sideeffect "", "{rsi}"(i32 %2) nounwind
// CHECK:  %3 = call i32 asm sideeffect "", "={rsi}"() nounwind
// CHECK:  call void asm sideeffect "; $0 This asm uses rsi", "{rsi},~{dirflag},~{fpsr},~{flags}"(i32 %3)

  return a;
// CHECK:  %4 = load i32* %a, align 4
// CHECK:  call void asm sideeffect "", "{rsi}"(i32 %4) nounwind
// CHECK:  %5 = call i32 asm sideeffect "", "={rsi}"() nounwind
// CHECK:  store i32 %5, i32* %0, align 4
// CHECK:  %6 = load i32* %0, align 4
// CHECK:  store i32 %6, i32* %retval, align 4
}
