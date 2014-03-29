; RUN: llc < %s -march=arm64 | FileCheck %s

define i8* @t() nounwind {
entry:
; CHECK-LABEL: t:
; CHECK: stp fp, lr, [sp, #-16]!
; CHECK: mov fp, sp
; CHECK: mov x0, fp
; CHECK: ldp fp, lr, [sp], #16
; CHECK: ret
	%0 = call i8* @llvm.frameaddress(i32 0)
        ret i8* %0
}

declare i8* @llvm.frameaddress(i32) nounwind readnone
