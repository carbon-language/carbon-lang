; RUN: llc < %s -march=bfin -verify-machineinstrs | FileCheck %s

; CHECK: .section .rodata
; CHECK: JTI0_0:
; CHECK: .long .BB0_1

define i32 @oper(i32 %op, i32 %A, i32 %B) {
entry:
        switch i32 %op, label %bbx [
               i32 1 , label %bb1
               i32 2 , label %bb2
               i32 3 , label %bb3
               i32 4 , label %bb4
               i32 5 , label %bb5
               i32 6 , label %bb6
               i32 7 , label %bb7
               i32 8 , label %bb8
               i32 9 , label %bb9
               i32 10, label %bb10
        ]
bb1:
	%R1 = add i32 %A, %B		; <i32> [#uses=1]
	ret i32 %R1
bb2:
	%R2 = sub i32 %A, %B		; <i32> [#uses=1]
	ret i32 %R2
bb3:
	%R3 = mul i32 %A, %B		; <i32> [#uses=1]
	ret i32 %R3
bb4:
	%R4 = sdiv i32 %A, %B		; <i32> [#uses=1]
	ret i32 %R4
bb5:
	%R5 = udiv i32 %A, %B		; <i32> [#uses=1]
	ret i32 %R5
bb6:
	%R6 = srem i32 %A, %B		; <i32> [#uses=1]
	ret i32 %R6
bb7:
	%R7 = urem i32 %A, %B		; <i32> [#uses=1]
	ret i32 %R7
bb8:
	%R8 = and i32 %A, %B		; <i32> [#uses=1]
	ret i32 %R8
bb9:
	%R9 = or i32 %A, %B		; <i32> [#uses=1]
	ret i32 %R9
bb10:
	%R10 = xor i32 %A, %B		; <i32> [#uses=1]
	ret i32 %R10
bbx:
        ret i32 0
}
