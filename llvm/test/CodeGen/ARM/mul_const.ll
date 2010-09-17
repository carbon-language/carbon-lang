; RUN: llc < %s -march=arm | FileCheck %s

define i32 @t9(i32 %v) nounwind readnone {
entry:
; CHECK: t9:
; CHECK: add r0, r0, r0, lsl #3
	%0 = mul i32 %v, 9
	ret i32 %0
}

define i32 @t7(i32 %v) nounwind readnone {
entry:
; CHECK: t7:
; CHECK: rsb r0, r0, r0, lsl #3
	%0 = mul i32 %v, 7
	ret i32 %0
}

define i32 @t5(i32 %v) nounwind readnone {
entry:
; CHECK: t5:
; CHECK: add r0, r0, r0, lsl #2
        %0 = mul i32 %v, 5
        ret i32 %0
}

define i32 @t3(i32 %v) nounwind readnone {
entry:
; CHECK: t3:
; CHECK: add r0, r0, r0, lsl #1
        %0 = mul i32 %v, 3
        ret i32 %0
}

define i32 @t12288(i32 %v) nounwind readnone {
entry:
; CHECK: t12288:
; CHECK: add r0, r0, r0, lsl #1
; CHECK: lsl{{.*}}#12
        %0 = mul i32 %v, 12288
        ret i32 %0
}

