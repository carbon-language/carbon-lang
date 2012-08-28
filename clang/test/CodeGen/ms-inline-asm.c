// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -O0 -fms-extensions -fenable-experimental-ms-inline-asm -w -emit-llvm -o - | FileCheck %s

void t1() {
// CHECK: @t1
// CHECK: call void asm sideeffect "", "~{dirflag},~{fpsr},~{flags}"() nounwind ia_nsdialect
// CHECK: ret void
  __asm {}
}

void t2() {
// CHECK: @t2
// CHECK: call void asm sideeffect "nop", "~{dirflag},~{fpsr},~{flags}"() nounwind ia_nsdialect
// CHECK: call void asm sideeffect "nop", "~{dirflag},~{fpsr},~{flags}"() nounwind ia_nsdialect
// CHECK: call void asm sideeffect "nop", "~{dirflag},~{fpsr},~{flags}"() nounwind ia_nsdialect
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
// CHECK: call void asm sideeffect "mov ebx, eax", "~{ebx},~{dirflag},~{fpsr},~{flags}"() nounwind ia_nsdialect
// CHECK: call void asm sideeffect "mov ecx, ebx", "~{ecx},~{dirflag},~{fpsr},~{flags}"() nounwind ia_nsdialect
// CHECK: ret void
  __asm mov ebx, eax
  __asm mov ecx, ebx
}

void t5(void) {
// CHECK: @t5
// CHECK: call void asm sideeffect "mov ebx, eax\0Amov ecx, ebx", "~{ebx},~{ecx},~{dirflag},~{fpsr},~{flags}"() nounwind ia_nsdialect
// CHECK: ret void
  __asm mov ebx, eax __asm mov ecx, ebx
}

void t6(void) {
  __asm int 0x2c
// CHECK: t6
// CHECK: call void asm sideeffect "int 0x2c", "~{dirflag},~{fpsr},~{flags}"() nounwind ia_nsdialect
}

void t7() {
  __asm {
    int 0x2c ; } asm comments are fun! }{
  }
  __asm {}
// CHECK: t7
// CHECK: call void asm sideeffect "int 0x2c", "~{dirflag},~{fpsr},~{flags}"() nounwind ia_nsdialect
// CHECK: call void asm sideeffect "", "~{dirflag},~{fpsr},~{flags}"() nounwind ia_nsdialect
}
int t8() {
  __asm int 3 ; } comments for single-line asm
  __asm {}
  __asm int 4
  return 10;
// CHECK: t8
// CHECK: call void asm sideeffect "int 3", "~{dirflag},~{fpsr},~{flags}"() nounwind ia_nsdialect
// CHECK: call void asm sideeffect "", "~{dirflag},~{fpsr},~{flags}"() nounwind ia_nsdialect
// CHECK: call void asm sideeffect "int 4", "~{dirflag},~{fpsr},~{flags}"() nounwind ia_nsdialect
// CHECK: ret i32 10
}
void t9() {
  __asm {
    push ebx
    mov ebx, 0x07
    pop ebx
  }
// CHECK: t9
// CHECK: call void asm sideeffect "push ebx\0Amov ebx, 0x07\0Apop ebx", "~{ebx},~{dirflag},~{fpsr},~{flags}"() nounwind ia_nsdialect
}

unsigned t10(void) {
  unsigned i = 1, j;
  __asm {
    mov eax, i
    mov j, eax
  }
  return j;
// CHECK: t10
// CHECK: [[I:%[a-zA-Z0-9]+]] = alloca i32, align 4
// CHECK: [[J:%[a-zA-Z0-9]+]] = alloca i32, align 4
// CHECK: store i32 1, i32* [[I]], align 4
// Note: The AsmParser isn't properly matching the second instruction (i.e., j is being marked as an input, when in fact it is an output).
// CHECK: call void asm sideeffect "mov eax, i\0Amov j, eax", "r,r,~{eax},~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}}, i32 %{{.*}}) nounwind ia_nsdialect
// CHECK: [[RET:%[a-zA-Z0-9]+]] = load i32* [[J]], align 4
// CHECK: ret i32 [[RET]]
}

void t11(void) {
  __asm EVEN
  __asm ALIGN
}

void t12(void) {
  __asm {
    _emit 0x4A
    _emit 0x43
    _emit 0x4B
  }
}

void t13(void) {
  unsigned arr[10];
  __asm LENGTH arr ; sizeof(arr)/sizeof(arr[0])
  __asm SIZE arr   ; sizeof(arr)
  __asm TYPE arr   ; sizeof(arr[0])
}
