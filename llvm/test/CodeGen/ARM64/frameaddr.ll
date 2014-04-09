; RUN: llc < %s -march=arm64 | FileCheck %s

define i8* @t() nounwind {
entry:
; CHECK-LABEL: t:
; CHECK: stp x29, lr, [sp, #-16]!
; CHECK: mov x29, sp
; CHECK: mov x0, x29
; CHECK: ldp x29, lr, [sp], #16
; CHECK: ret
	%0 = call i8* @llvm.frameaddress(i32 0)
        ret i8* %0
}

declare i8* @llvm.frameaddress(i32) nounwind readnone
