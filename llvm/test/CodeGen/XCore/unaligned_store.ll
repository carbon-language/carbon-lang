; RUN: llc < %s -march=xcore | FileCheck %s

; Byte aligned store.
; CHECK: align1:
; CHECK: bl __misaligned_store
define void @align1(i32* %p, i32 %val) nounwind {
entry:
	store i32 %val, i32* %p, align 1
	ret void
}

; Half word aligned store.
; CHECK: align2
; CHECK: st16
; CHECK: st16
define void @align2(i32* %p, i32 %val) nounwind {
entry:
	store i32 %val, i32* %p, align 2
	ret void
}
