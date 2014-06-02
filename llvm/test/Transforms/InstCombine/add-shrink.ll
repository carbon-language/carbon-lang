; RUN: opt < %s -instcombine -S | FileCheck %s

; CHECK-LABEL: define i64 @test
define i64 @test1(i32 %A) {
; CHECK: %[[ADD:.*]] = add nsw i32 %B, %C
; CHECK: %F = sext i32 %[[ADD]] to i64
; CHECK: ret i64 %F

	%B = ashr i32 %A, 7		; <i32> [#uses=1]
	%C = ashr i32 %A, 9		; <i32> [#uses=1]
	%D = sext i32 %B to i64		; <i64> [#uses=1]
	%E = sext i32 %C to i64		; <i64> [#uses=1]
	%F = add i64 %D, %E		; <i64> [#uses=1]
	ret i64 %F
}

