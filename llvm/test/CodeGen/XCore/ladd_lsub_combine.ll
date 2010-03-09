; RUN: llvm-as < %s | llc -march=xcore | FileCheck %s

; Only needs one ladd
define i64 @f1(i32 %x, i32 %y) nounwind {
entry:
	%0 = zext i32 %x to i64		; <i64> [#uses=1]
	%1 = zext i32 %y to i64		; <i64> [#uses=1]
	%2 = add i64 %1, %0		; <i64> [#uses=1]
	ret i64 %2
}
; CHECK: f1:
; CHECK: ldc r2, 0
; CHECK-NEXT: ladd r1, r0, r1, r0, r2
; CHECK-NEXT: retsp 0

; Only needs one lsub and one neg
define i64 @f2(i32 %x, i32 %y) nounwind {
entry:
	%0 = zext i32 %x to i64		; <i64> [#uses=1]
	%1 = zext i32 %y to i64		; <i64> [#uses=1]
	%2 = sub i64 %1, %0		; <i64> [#uses=1]
	ret i64 %2
}
; CHECK: f2:
; CHECK: ldc r2, 0
; CHECK-NEXT: lsub r1, r0, r1, r0, r2
; CHECK-NEXT: neg r1, r1
; CHECK-NEXT: retsp 0
