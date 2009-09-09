; RUN: llc < %s

target datalayout = "E-p:64:64:64-i8:8:16-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-a0:16:16"
target triple = "s390x-linux"

define void @compdecomp(i8* nocapture %data, i64 %data_len) nounwind {
entry:
	br label %for.body38

for.body38:		; preds = %for.body38, %entry
	br i1 undef, label %for.cond220, label %for.body38

for.cond220:		; preds = %for.cond220, %for.body38
	br i1 false, label %for.cond220, label %for.end297

for.end297:		; preds = %for.cond220
	%tmp334 = load i8* undef		; <i8> [#uses=1]
	%conv343 = zext i8 %tmp334 to i32		; <i32> [#uses=1]
	%sub344 = add i32 %conv343, -1		; <i32> [#uses=1]
	%shl345 = shl i32 1, %sub344		; <i32> [#uses=1]
	%conv346 = sext i32 %shl345 to i64		; <i64> [#uses=1]
	br label %for.body356

for.body356:		; preds = %for.body356, %for.end297
	%mask.1633 = phi i64 [ %conv346, %for.end297 ], [ undef, %for.body356 ]		; <i64> [#uses=0]
	br label %for.body356
}
