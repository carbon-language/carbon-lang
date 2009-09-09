; RUN: llc < %s | not grep aghi
; RUN: llc < %s | FileCheck %s

target datalayout = "E-p:64:64:64-i1:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128"
target triple = "s390x-unknown-linux-gnu"

define void @foo1(i64* nocapture %a, i64 %idx, i64 %val) nounwind {
entry:

; CHECK: foo1:
; CHECK:   stg %r4, 8(%r1,%r2)
	%add.ptr.sum = add i64 %idx, 1		; <i64> [#uses=1]
	%add.ptr2 = getelementptr i64* %a, i64 %add.ptr.sum		; <i64*> [#uses=1]
	store i64 %val, i64* %add.ptr2
	ret void
}

define void @foo2(i32* nocapture %a, i64 %idx, i32 %val) nounwind {
entry:
; CHECK: foo2:
; CHECK:   st %r4, 4(%r1,%r2)
	%add.ptr.sum = add i64 %idx, 1		; <i64> [#uses=1]
	%add.ptr2 = getelementptr i32* %a, i64 %add.ptr.sum		; <i32*> [#uses=1]
	store i32 %val, i32* %add.ptr2
	ret void
}

define void @foo3(i16* nocapture %a, i64 %idx, i16 zeroext %val) nounwind {
entry:
; CHECK: foo3:
; CHECK: sth     %r4, 2(%r1,%r2)
	%add.ptr.sum = add i64 %idx, 1		; <i64> [#uses=1]
	%add.ptr2 = getelementptr i16* %a, i64 %add.ptr.sum		; <i16*> [#uses=1]
	store i16 %val, i16* %add.ptr2
	ret void
}

define void @foo4(i8* nocapture %a, i64 %idx, i8 zeroext %val) nounwind {
entry:
; CHECK: foo4:
; CHECK: stc     %r4, 1(%r3,%r2)
	%add.ptr.sum = add i64 %idx, 1		; <i64> [#uses=1]
	%add.ptr2 = getelementptr i8* %a, i64 %add.ptr.sum		; <i8*> [#uses=1]
	store i8 %val, i8* %add.ptr2
	ret void
}

define void @foo5(i8* nocapture %a, i64 %idx, i64 %val) nounwind {
entry:
; CHECK: foo5:
; CHECK: stc     %r4, 1(%r3,%r2)
	%add.ptr.sum = add i64 %idx, 1		; <i64> [#uses=1]
	%add.ptr2 = getelementptr i8* %a, i64 %add.ptr.sum		; <i8*> [#uses=1]
	%conv = trunc i64 %val to i8		; <i8> [#uses=1]
	store i8 %conv, i8* %add.ptr2
	ret void
}

define void @foo6(i16* nocapture %a, i64 %idx, i64 %val) nounwind {
entry:
; CHECK: foo6:
; CHECK: sth     %r4, 2(%r1,%r2)
	%add.ptr.sum = add i64 %idx, 1		; <i64> [#uses=1]
	%add.ptr2 = getelementptr i16* %a, i64 %add.ptr.sum		; <i16*> [#uses=1]
	%conv = trunc i64 %val to i16		; <i16> [#uses=1]
	store i16 %conv, i16* %add.ptr2
	ret void
}

define void @foo7(i32* nocapture %a, i64 %idx, i64 %val) nounwind {
entry:
; CHECK: foo7:
; CHECK: st      %r4, 4(%r1,%r2)
	%add.ptr.sum = add i64 %idx, 1		; <i64> [#uses=1]
	%add.ptr2 = getelementptr i32* %a, i64 %add.ptr.sum		; <i32*> [#uses=1]
	%conv = trunc i64 %val to i32		; <i32> [#uses=1]
	store i32 %conv, i32* %add.ptr2
	ret void
}
