; With reassociation, constant folding can eliminate the 12 and -12 constants.
;
; RUN: opt < %s -reassociate  -gvn -instcombine -S | FileCheck %s

define i32 @test1(i32 %arg) {
	%tmp1 = sub i32 -12, %arg
	%tmp2 = add i32 %tmp1, 12
	ret i32 %tmp2
; CHECK: @test1
; CHECK-NEXT: sub i32 0, %arg
; CHECK-NEXT: ret i32
}

define i32 @test2(i32 %reg109, i32 %reg1111) {
	%reg115 = add i32 %reg109, -30		; <i32> [#uses=1]
	%reg116 = add i32 %reg115, %reg1111		; <i32> [#uses=1]
	%reg117 = add i32 %reg116, 30		; <i32> [#uses=1]
	ret i32 %reg117
; CHECK: @test2
; CHECK-NEXT: add i32 %reg1111, %reg109
; CHECK-NEXT: ret i32
}

@e = external global i32		; <i32*> [#uses=3]
@a = external global i32		; <i32*> [#uses=3]
@b = external global i32		; <i32*> [#uses=3]
@c = external global i32		; <i32*> [#uses=3]
@f = external global i32		; <i32*> [#uses=3]

define void @test3() {
	%A = load i32* @a		; <i32> [#uses=2]
	%B = load i32* @b		; <i32> [#uses=2]
	%C = load i32* @c		; <i32> [#uses=2]
	%t1 = add i32 %A, %B		; <i32> [#uses=1]
	%t2 = add i32 %t1, %C		; <i32> [#uses=1]
	%t3 = add i32 %C, %A		; <i32> [#uses=1]
	%t4 = add i32 %t3, %B		; <i32> [#uses=1]
	; e = (a+b)+c;
        store i32 %t2, i32* @e
        ; f = (a+c)+b
	store i32 %t4, i32* @f
	ret void
; CHECK: @test3
; CHECK: add i32
; CHECK: add i32
; CHECK-NOT: add i32
; CHECK: ret void
}

define void @test4() {
	%A = load i32* @a		; <i32> [#uses=2]
	%B = load i32* @b		; <i32> [#uses=2]
	%C = load i32* @c		; <i32> [#uses=2]
	%t1 = add i32 %A, %B		; <i32> [#uses=1]
	%t2 = add i32 %t1, %C		; <i32> [#uses=1]
	%t3 = add i32 %C, %A		; <i32> [#uses=1]
	%t4 = add i32 %t3, %B		; <i32> [#uses=1]
	; e = c+(a+b)
        store i32 %t2, i32* @e
        ; f = (c+a)+b
	store i32 %t4, i32* @f
	ret void
; CHECK: @test4
; CHECK: add i32
; CHECK: add i32
; CHECK-NOT: add i32
; CHECK: ret void
}

define void @test5() {
	%A = load i32* @a		; <i32> [#uses=2]
	%B = load i32* @b		; <i32> [#uses=2]
	%C = load i32* @c		; <i32> [#uses=2]
	%t1 = add i32 %B, %A		; <i32> [#uses=1]
	%t2 = add i32 %t1, %C		; <i32> [#uses=1]
	%t3 = add i32 %C, %A		; <i32> [#uses=1]
	%t4 = add i32 %t3, %B		; <i32> [#uses=1]
	; e = c+(b+a)
        store i32 %t2, i32* @e
        ; f = (c+a)+b
	store i32 %t4, i32* @f
	ret void
; CHECK: @test5
; CHECK: add i32
; CHECK: add i32
; CHECK-NOT: add i32
; CHECK: ret void
}

define i32 @test6() {
	%tmp.0 = load i32* @a		; <i32> [#uses=2]
	%tmp.1 = load i32* @b		; <i32> [#uses=2]
        ; (a+b)
	%tmp.2 = add i32 %tmp.0, %tmp.1		; <i32> [#uses=1]
	%tmp.4 = load i32* @c		; <i32> [#uses=2]
	; (a+b)+c
        %tmp.5 = add i32 %tmp.2, %tmp.4		; <i32> [#uses=1]
	; (a+c)
        %tmp.8 = add i32 %tmp.0, %tmp.4		; <i32> [#uses=1]
	; (a+c)+b
        %tmp.11 = add i32 %tmp.8, %tmp.1		; <i32> [#uses=1]
	; X ^ X = 0
        %RV = xor i32 %tmp.5, %tmp.11		; <i32> [#uses=1]
	ret i32 %RV
; CHECK: @test6
; CHECK: ret i32 0
}
