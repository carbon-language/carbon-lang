; RUN: llc < %s -march=pic16 | FileCheck %s
; XFAIL: vg_leak

target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8-f32:32:32"
target triple = "pic16-"
@i = global i32 -10, align 1		; <i32*> [#uses=1]
@j = global i32 -20, align 1		; <i32*> [#uses=1]
@pc = global i8* inttoptr (i64 160 to i8*), align 1		; <i8**> [#uses=3]
@main.auto.k = internal global i32 0		; <i32*> [#uses=2]

define void @main() nounwind {
entry:
	%tmp = load i32* @i		; <i32> [#uses=1]
	%tmp1 = load i32* @j		; <i32> [#uses=1]
	%add = add i32 %tmp, %tmp1		; <i32> [#uses=1]
	store i32 %add, i32* @main.auto.k
	%tmp2 = load i32* @main.auto.k		; <i32> [#uses=1]
	%add3 = add i32 %tmp2, 32		; <i32> [#uses=1]
	%conv = trunc i32 %add3 to i8		; <i8> [#uses=1]
	%tmp4 = load i8** @pc		; <i8*> [#uses=1]
	store i8 %conv, i8* %tmp4
	%tmp5 = load i8** @pc		; <i8*> [#uses=1]
	%tmp6 = load i8* %tmp5		; <i8> [#uses=1]
	%conv7 = sext i8 %tmp6 to i16		; <i16> [#uses=1]
	%sub = sub i16 %conv7, 1		; <i16> [#uses=1]
	%conv8 = trunc i16 %sub to i8		; <i8> [#uses=1]
	%tmp9 = load i8** @pc		; <i8*> [#uses=1]
	store i8 %conv8, i8* %tmp9
	ret void
}

; CHECK: movf @i + 0, W
