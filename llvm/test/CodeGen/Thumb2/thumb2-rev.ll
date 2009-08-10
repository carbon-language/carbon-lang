; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2,+v7a | FileCheck %s

define i32 @f1(i32 %a) {
; CHECK: f1:
; CHECK: rev r0, r0
    %tmp = tail call i32 @llvm.bswap.i32(i32 %a)
    ret i32 %tmp
}

declare i32 @llvm.bswap.i32(i32) nounwind readnone
