; RUN: llc < %s

target datalayout = "E-p:64:64:64-i1:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128"
target triple = "s390x-unknown-linux-gnu"

define i32 @main() nounwind {
entry:
	%call = call i32 (...)* @random() nounwind		; <i32> [#uses=0]
	unreachable
}

declare i32 @random(...)
