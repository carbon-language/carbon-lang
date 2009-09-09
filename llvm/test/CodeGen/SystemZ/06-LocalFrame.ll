; RUN: llc < %s | grep 160 | count 1
; RUN: llc < %s | grep 328 | count 1
; RUN: llc < %s | grep 168 | count 1

target datalayout = "E-p:64:64:64-i1:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128"
target triple = "s390x-unknown-linux-gnu"

define noalias i64* @foo(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f) nounwind readnone {
entry:
	%g = alloca i64, align 8		; <i64*> [#uses=1]
	%add.ptr = getelementptr i64* %g, i64 %f		; <i64*> [#uses=1]
	ret i64* %add.ptr
}
