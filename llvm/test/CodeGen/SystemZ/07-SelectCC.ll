; RUN: llc < %s | grep clgr

target datalayout = "E-p:64:64:64-i1:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128"
target triple = "s390x-unknown-linux-gnu"

define i64 @foo(i64 %a, i64 %b) nounwind readnone {
entry:
	%cmp = icmp ult i64 %a, %b		; <i1> [#uses=1]
	%cond = select i1 %cmp, i64 %a, i64 %b		; <i64> [#uses=1]
	ret i64 %cond
}
