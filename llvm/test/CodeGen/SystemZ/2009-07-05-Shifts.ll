; RUN: llc < %s

target datalayout = "E-p:64:64:64-i8:8:16-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-a0:16:16"
target triple = "s390x-linux"

define signext i32 @bit_place_piece(i32 signext %col, i32 signext %player, i64* nocapture %b1, i64* nocapture %b2) nounwind {
entry:
	br i1 undef, label %for.body, label %return

for.body:		; preds = %entry
	%add = add i32 0, %col		; <i32> [#uses=1]
	%sh_prom = zext i32 %add to i64		; <i64> [#uses=1]
	%shl = shl i64 1, %sh_prom		; <i64> [#uses=1]
	br i1 undef, label %if.then13, label %if.else

if.then13:		; preds = %for.body
	ret i32 0

if.else:		; preds = %for.body
	%or34 = or i64 undef, %shl		; <i64> [#uses=0]
	ret i32 0

return:		; preds = %entry
	ret i32 1
}
