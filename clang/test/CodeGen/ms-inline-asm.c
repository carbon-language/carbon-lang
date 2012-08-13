// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -O0 -fms-extensions -fenable-experimental-ms-inline-asm -w -emit-llvm -o - | FileCheck %s

void t1() {
// CHECK: @t1
// CHECK: call void asm sideeffect "", "~{dirflag},~{fpsr},~{flags}"() nounwind ia_nsdialect
// CHECK: ret void
  __asm {}
}

void t2() {
// CHECK: @t2
// CHECK: call void asm sideeffect "nop\0Anop\0Anop", "~{dirflag},~{fpsr},~{flags}"() nounwind ia_nsdialect
// CHECK: ret void
  __asm nop
  __asm nop
  __asm nop
}

void t3() {
// CHECK: @t3
// CHECK: call void asm sideeffect "nop\0Anop\0Anop", "~{dirflag},~{fpsr},~{flags}"() nounwind ia_nsdialect
// CHECK: ret void
  __asm nop __asm nop __asm nop
}

void t4(void) {
// CHECK: @t4
// CHECK: call void asm sideeffect "mov ebx, eax\0Amov ecx, ebx", "~{dirflag},~{fpsr},~{flags}"() nounwind ia_nsdialect
// CHECK: ret void
  __asm mov ebx, eax
  __asm mov ecx, ebx
}

void t5(void) {
// CHECK: @t5
// CHECK: call void asm sideeffect "mov ebx, eax\0Amov ecx, ebx", "~{dirflag},~{fpsr},~{flags}"() nounwind ia_nsdialect
// CHECK: ret void
  __asm mov ebx, eax __asm mov ecx, ebx
}

