; RUN: llc < %s -march=systemz | not grep aghi
; RUN: llc < %s -march=systemz | grep llgf | count 1
; RUN: llc < %s -march=systemz | grep llgh | count 1
; RUN: llc < %s -march=systemz | grep llgc | count 1
; RUN: llc < %s -march=systemz | grep lgf  | count 2
; RUN: llc < %s -march=systemz | grep lgh  | count 2
; RUN: llc < %s -march=systemz | grep lgb  | count 1


target datalayout = "E-p:64:64:64-i1:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128"
target triple = "s390x-unknown-linux-gnu"

define zeroext i64 @foo1(i64* nocapture %a, i64 %idx) nounwind readonly {
entry:
	%add.ptr.sum = add i64 %idx, 1		; <i64> [#uses=1]
	%add.ptr2 = getelementptr i64* %a, i64 %add.ptr.sum		; <i64*> [#uses=1]
	%tmp3 = load i64* %add.ptr2		; <i64> [#uses=1]
	ret i64 %tmp3
}

define zeroext i32 @foo2(i32* nocapture %a, i64 %idx) nounwind readonly {
entry:
	%add.ptr.sum = add i64 %idx, 1		; <i64> [#uses=1]
	%add.ptr2 = getelementptr i32* %a, i64 %add.ptr.sum		; <i32*> [#uses=1]
	%tmp3 = load i32* %add.ptr2		; <i32> [#uses=1]
	ret i32 %tmp3
}

define zeroext i16 @foo3(i16* nocapture %a, i64 %idx) nounwind readonly {
entry:
	%add.ptr.sum = add i64 %idx, 1		; <i64> [#uses=1]
	%add.ptr2 = getelementptr i16* %a, i64 %add.ptr.sum		; <i16*> [#uses=1]
	%tmp3 = load i16* %add.ptr2		; <i16> [#uses=1]
	ret i16 %tmp3
}

define zeroext i8 @foo4(i8* nocapture %a, i64 %idx) nounwind readonly {
entry:
	%add.ptr.sum = add i64 %idx, 1		; <i64> [#uses=1]
	%add.ptr2 = getelementptr i8* %a, i64 %add.ptr.sum		; <i8*> [#uses=1]
	%tmp3 = load i8* %add.ptr2		; <i8> [#uses=1]
	ret i8 %tmp3
}

define signext i64 @foo5(i64* nocapture %a, i64 %idx) nounwind readonly {
entry:
	%add.ptr.sum = add i64 %idx, 1		; <i64> [#uses=1]
	%add.ptr2 = getelementptr i64* %a, i64 %add.ptr.sum		; <i64*> [#uses=1]
	%tmp3 = load i64* %add.ptr2		; <i64> [#uses=1]
	ret i64 %tmp3
}

define signext i32 @foo6(i32* nocapture %a, i64 %idx) nounwind readonly {
entry:
	%add.ptr.sum = add i64 %idx, 1		; <i64> [#uses=1]
	%add.ptr2 = getelementptr i32* %a, i64 %add.ptr.sum		; <i32*> [#uses=1]
	%tmp3 = load i32* %add.ptr2		; <i32> [#uses=1]
	ret i32 %tmp3
}

define signext i16 @foo7(i16* nocapture %a, i64 %idx) nounwind readonly {
entry:
	%add.ptr.sum = add i64 %idx, 1		; <i64> [#uses=1]
	%add.ptr2 = getelementptr i16* %a, i64 %add.ptr.sum		; <i16*> [#uses=1]
	%tmp3 = load i16* %add.ptr2		; <i16> [#uses=1]
	ret i16 %tmp3
}

define signext i8 @foo8(i8* nocapture %a, i64 %idx) nounwind readonly {
entry:
	%add.ptr.sum = add i64 %idx, 1		; <i64> [#uses=1]
	%add.ptr2 = getelementptr i8* %a, i64 %add.ptr.sum		; <i8*> [#uses=1]
	%tmp3 = load i8* %add.ptr2		; <i8> [#uses=1]
	ret i8 %tmp3
}
