; RUN: llc < %s | grep msgr | count 2
; RUN: llc < %s | grep msr  | count 2

target datalayout = "E-p:64:64:64-i1:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128"
target triple = "s390x-unknown-linux-gnu"

define i64 @foo(i64 %a, i64 %b) nounwind readnone {
entry:
	%mul = mul i64 %b, %a		; <i64> [#uses=1]
	ret i64 %mul
}

define i64 @foo2(i64 %a, i64 %b) nounwind readnone {
entry:
	%mul = mul i64 %b, %a		; <i64> [#uses=1]
	ret i64 %mul
}

define i32 @foo3(i32 %a, i32 %b) nounwind readnone {
entry:
	%mul = mul i32 %b, %a		; <i32> [#uses=1]
	ret i32 %mul
}

define i32 @foo4(i32 %a, i32 %b) nounwind readnone {
entry:
	%mul = mul i32 %b, %a		; <i32> [#uses=1]
	ret i32 %mul
}
