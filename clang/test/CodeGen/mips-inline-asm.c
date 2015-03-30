// REQUIRES: mips-registered-target
// RUN: %clang_cc1 -triple mips-linux-gnu -emit-llvm -o - %s | FileCheck %s

int data;

void m () {
  asm("lw $1, %0" :: "m"(data));
  // CHECK: call void asm sideeffect "lw $$1, $0", "*m,~{$1}"(i32* @data)
}

void ZC () {
  asm("ll $1, %0" :: "ZC"(data));
  // CHECK: call void asm sideeffect "ll $$1, $0", "*^ZC,~{$1}"(i32* @data)
}

void R () {
  asm("lw $1, %0" :: "R"(data));
  // CHECK: call void asm sideeffect "lw $$1, $0", "*R,~{$1}"(i32* @data)
}
