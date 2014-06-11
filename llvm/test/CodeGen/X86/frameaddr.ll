; RUN: llc < %s -march=x86                                | FileCheck %s --check-prefix=CHECK-32
; RUN: llc < %s -march=x86    -fast-isel -fast-isel-abort | FileCheck %s --check-prefix=CHECK-32
; RUN: llc < %s -march=x86-64                             | FileCheck %s --check-prefix=CHECK-64
; RUN: llc < %s -march=x86-64 -fast-isel -fast-isel-abort | FileCheck %s --check-prefix=CHECK-64

define i8* @test1() nounwind {
entry:
; CHECK-32:      movl %esp, %ebp
; CHECK-32-NEXT: movl %ebp, %eax
; CHECK-64:      movq %rsp, %rbp
; CHECK-64-NEXT: movq %rbp, %rax
  %0 = tail call i8* @llvm.frameaddress(i32 0)
  ret i8* %0
}

define i8* @test2() nounwind {
entry:
; CHECK-32:      movl %esp, %ebp
; CHECK-32-NEXT: movl (%ebp), %eax
; CHECK-32-NEXT: movl (%eax), %eax
; CHECK-64:      movq %rsp, %rbp
; CHECK-64-NEXT: movq (%rbp), %rax
; CHECK-64-NEXT: movq (%rax), %rax
  %0 = tail call i8* @llvm.frameaddress(i32 2)
  ret i8* %0
}
declare i8* @llvm.frameaddress(i32) nounwind readnone
