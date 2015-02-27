; RUN: llc < %s -march=xcore | FileCheck %s

; Byte aligned load.
; CHECK: align1
; CHECK: bl __misaligned_load
define i32 @align1(i32* %p) nounwind {
entry:
	%0 = load i32, i32* %p, align 1		; <i32> [#uses=1]
	ret i32 %0
}

; Half word aligned load.
; CHECK-LABEL: align2:
; CHECK: ld16s
; CHECK: ld16s
; CHECK: or
define i32 @align2(i32* %p) nounwind {
entry:
	%0 = load i32, i32* %p, align 2		; <i32> [#uses=1]
	ret i32 %0
}

@a = global [5 x i8] zeroinitializer, align 4

; Constant offset from word aligned base.
; CHECK-LABEL: align3:
; CHECK: ldw {{r[0-9]+}}, dp
; CHECK: ldw {{r[0-9]+}}, dp
; CHECK: or
define i32 @align3() nounwind {
entry:
	%0 = load i32, i32* bitcast (i8* getelementptr ([5 x i8]* @a, i32 0, i32 1) to i32*), align 1
	ret i32 %0
}
