; RUN: llc < %s | grep dsgr  | count 2
; RUN: llc < %s | grep dsgfr | count 2
; RUN: llc < %s | grep dlr   | count 2
; RUN: llc < %s | grep dlgr  | count 2

target datalayout = "E-p:64:64:64-i1:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128"
target triple = "s390x-unknown-linux-gnu"

define i64 @div(i64 %a, i64 %b) nounwind readnone {
entry:
	%div = sdiv i64 %a, %b		; <i64> [#uses=1]
	ret i64 %div
}

define i32 @div1(i32 %a, i32 %b) nounwind readnone {
entry:
	%div = sdiv i32 %a, %b		; <i32> [#uses=1]
	ret i32 %div
}

define i64 @div2(i64 %a, i64 %b) nounwind readnone {
entry:
	%div = udiv i64 %a, %b		; <i64> [#uses=1]
	ret i64 %div
}

define i32 @div3(i32 %a, i32 %b) nounwind readnone {
entry:
	%div = udiv i32 %a, %b		; <i32> [#uses=1]
	ret i32 %div
}

define i64 @rem(i64 %a, i64 %b) nounwind readnone {
entry:
	%rem = srem i64 %a, %b		; <i64> [#uses=1]
	ret i64 %rem
}

define i32 @rem1(i32 %a, i32 %b) nounwind readnone {
entry:
	%rem = srem i32 %a, %b		; <i32> [#uses=1]
	ret i32 %rem
}

define i64 @rem2(i64 %a, i64 %b) nounwind readnone {
entry:
	%rem = urem i64 %a, %b		; <i64> [#uses=1]
	ret i64 %rem
}

define i32 @rem3(i32 %a, i32 %b) nounwind readnone {
entry:
	%rem = urem i32 %a, %b		; <i32> [#uses=1]
	ret i32 %rem
}
