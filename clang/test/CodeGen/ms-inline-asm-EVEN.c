// RUN: %clang_cc1 %s -triple i386-unknown-unknown -fasm-blocks -emit-llvm -o - | FileCheck %s

// CHECK: .byte 64
// CHECK: .byte 64
// CHECK: .byte 64
// CHECK:  .even
void t1() {
  __asm {
    .byte 64
    .byte 64
    .byte 64
    EVEN
    mov eax, ebx
  }
}
