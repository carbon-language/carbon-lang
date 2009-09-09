; RUN: llc < %s -mattr=+z10 | grep mvghi | count 1
; RUN: llc < %s -mattr=+z10 | grep mvhi  | count 1
; RUN: llc < %s -mattr=+z10 | grep mvhhi | count 1
; RUN: llc < %s | grep mvi   | count 2
; RUN: llc < %s | grep mviy  | count 1

target datalayout = "E-p:64:64:64-i1:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128"
target triple = "s390x-unknown-linux-gnu"

define void @foo1(i64* nocapture %a, i64 %idx) nounwind {
entry:
	%add.ptr = getelementptr i64* %a, i64 1		; <i64*> [#uses=1]
	store i64 1, i64* %add.ptr
	ret void
}

define void @foo2(i32* nocapture %a, i64 %idx) nounwind {
entry:
	%add.ptr = getelementptr i32* %a, i64 1		; <i32*> [#uses=1]
	store i32 2, i32* %add.ptr
	ret void
}

define void @foo3(i16* nocapture %a, i64 %idx) nounwind {
entry:
	%add.ptr = getelementptr i16* %a, i64 1		; <i16*> [#uses=1]
	store i16 3, i16* %add.ptr
	ret void
}

define void @foo4(i8* nocapture %a, i64 %idx) nounwind {
entry:
	%add.ptr = getelementptr i8* %a, i64 1		; <i8*> [#uses=1]
	store i8 4, i8* %add.ptr
	ret void
}

define void @foo5(i8* nocapture %a, i64 %idx) nounwind {
entry:
        %add.ptr = getelementptr i8* %a, i64 -1         ; <i8*> [#uses=1]
        store i8 4, i8* %add.ptr
        ret void
}

define void @foo6(i16* nocapture %a, i64 %idx) nounwind {
entry:
        %add.ptr = getelementptr i16* %a, i64 -1         ; <i16*> [#uses=1]
        store i16 3, i16* %add.ptr
        ret void
}
