; RUN: llc < %s -march=systemz | grep rll

target datalayout = "E-p:64:64:64-i8:8:16-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-a0:16:16"
target triple = "s390x-linux"

define i32 @rotl(i32 %x, i32 %y, i32 %z) nounwind readnone {
entry:
	%shl = shl i32 %x, 1		; <i32> [#uses=1]
	%sub = sub i32 32, 1		; <i32> [#uses=1]
	%shr = lshr i32 %x, %sub		; <i32> [#uses=1]
	%or = or i32 %shr, %shl		; <i32> [#uses=1]
	ret i32 %or
}
