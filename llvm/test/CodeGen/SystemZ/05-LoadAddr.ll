; RUN: llc < %s | grep lay | count 1

target datalayout = "E-p:64:64:64-i1:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128"
target triple = "s390x-unknown-linux-gnu"

define i64* @foo(i64* %a, i64 %idx) nounwind readnone {
entry:
	%add.ptr.sum = add i64 %idx, 1		; <i64> [#uses=1]
	%add.ptr2 = getelementptr i64* %a, i64 %add.ptr.sum		; <i64*> [#uses=1]
	ret i64* %add.ptr2
}
