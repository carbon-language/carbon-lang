; RUN: llc -march=msp430 < %s | FileCheck %s
; XFAIL: *
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-generic-generic"
@foo = common global i16 0, align 2
@bar = common global i16 0, align 2

define void @mov() nounwind {
; CHECK: mov:
; CHECK: mov.w	&bar, &foo
        %1 = load i16* @bar
        store i16 %1, i16* @foo
        ret void
}

define void @add() nounwind {
; CHECK: add:
; CHECK: add.w	&bar, &foo
	%1 = load i16* @bar
	%2 = load i16* @foo
	%3 = add i16 %2, %1
	store i16 %3, i16* @foo
	ret void
}

define void @and() nounwind {
; CHECK: and:
; CHECK: and.w	&bar, &foo
	%1 = load i16* @bar
	%2 = load i16* @foo
	%3 = and i16 %2, %1
	store i16 %3, i16* @foo
	ret void
}

define void @bis() nounwind {
; CHECK: bis:
; CHECK: bis.w	&bar, &foo
	%1 = load i16* @bar
	%2 = load i16* @foo
	%3 = or i16 %2, %1
	store i16 %3, i16* @foo
	ret void
}

define void @xor() nounwind {
; CHECK: xor:
; CHECK: xor.w	&bar, &foo
	%1 = load i16* @bar
	%2 = load i16* @foo
	%3 = xor i16 %2, %1
	store i16 %3, i16* @foo
	ret void
}

