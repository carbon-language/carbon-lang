; Functions with frames > 256K bytes require a frame pointer to access the stack.
; At present, functions must be compiled using '-fno-omit-frame-pointer'.
; RUN: llc < %s -march=xcore -disable-fp-elim | FileCheck %s

declare void @f0(i32*)

; CHECK: .section .cp.rodata.cst4,"aMc",@progbits,4
; CHECK: .LCPI[[NUM:[0-9_]+]]:
; CHECK: .long   99999
; CHECK: .text
; CHECK-LABEL:f1
; CHECK:      entsp 65535
; CHECK-NEXT: extsp 34465
; CHECK-NEXT: stw r10, sp[1]
; CHECK-NEXT: ldaw r10, sp[0]
; CHECK-NEXT: ldw r1, cp[.LCPI[[NUM]]]
; CHECK-NEXT: ldaw r0, r10[r1]
; CHECK-NEXT: extsp 1
; CHECK-NEXT: bl f0
; CHECK-NEXT: ldaw sp, sp[1]
; CHECK-NEXT: set sp, r10
; CHECK-NEXT: ldw r10, sp[1]
; CHECK-NEXT: ldaw sp, sp[65535]
; CHECK-NEXT: retsp 34465
define void @f1() nounwind {
entry:
  %0 = alloca [99998 x i32]
  %1 = getelementptr inbounds [99998 x i32]* %0, i32 0, i32 99997
  call void @f0(i32* %1)
  ret void
}

; CHECK-LABEL:f2
; CHECK:      mkmsk  [[REG:r[0-9]+]], 15
; CHECK-NEXT: ldaw r0, r10{{\[}}[[REG]]{{\]}}
define void @f2() nounwind {
entry:
  %0 = alloca [32768 x i32]
  %1 = getelementptr inbounds [32768 x i32]* %0, i32 0, i32 32765
  call void @f0(i32* %1)
  ret void
}
