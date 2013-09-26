; RUN: llc < %s -march=thumb -mattr=+v7         | FileCheck %s -check-prefix=THUMB2
; TODO: This test case will be merged back into prefetch.ll when ARM mode issue is solved.

declare void @llvm.prefetch(i8*, i32, i32, i32) nounwind

define void @t6() {
entry:
;ARM: t6:
;ARM: pld [sp]
;ARM: pld [sp, #50]

;THUMB2: t6:
;THUMB2: pld [sp]
;THUMB2: pld [sp, #50]

%red = alloca [100 x i8], align 1
%0 = getelementptr inbounds [100 x i8]* %red, i32 0, i32 0
%1 = getelementptr inbounds [100 x i8]* %red, i32 0, i32 50
call void @llvm.prefetch(i8* %0, i32 0, i32 3, i32 1)
call void @llvm.prefetch(i8* %1, i32 0, i32 3, i32 1)
ret void
}
