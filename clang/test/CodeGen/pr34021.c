// RUN: %clang_cc1 -fms-extensions %s -triple=i686-unknown-unknown -emit-llvm -o - | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 -fms-extensions %s -triple=x86_64-unknown-unknown -emit-llvm -o - | FileCheck %s --check-prefix=X64

typedef int v4si __attribute__ ((vector_size (16)));
v4si rep() {
// X86-LABEL: define <4 x i32> @rep
// X86:      %retval = alloca <4 x i32>, align 16
// X86-NEXT: %res = alloca <4 x i32>, align 16
// X86-NEXT: %0 = bitcast <4 x i32>* %retval to i128*
// X86-NEXT: %1 = call i64 asm sideeffect inteldialect "", "=A,~{dirflag},~{fpsr},~{flags}"()
// X86-NEXT: %2 = zext i64 %1 to i128
// X86-NEXT: store i128 %2, i128* %0, align 16
// X86-NEXT: %3 = load <4 x i32>, <4 x i32>* %res, align 16
// X86-NEXT: ret <4 x i32> %3
//
// X64-LABEL: define <4 x i32> @rep
// X64:      %res = alloca <4 x i32>, align 16
// X64-NEXT: call void asm sideeffect inteldialect "", "~{dirflag},~{fpsr},~{flags}"()
// X64-NEXT: %0 = load <4 x i32>, <4 x i32>* %res, align 16
// X64-NEXT: ret <4 x i32> %0
  v4si res;
  __asm {}
  return res;
}
