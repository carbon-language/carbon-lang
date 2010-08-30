; RUN: llc < %s

target datalayout = "E-p:64:64:64-i8:8:16-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-a0:16:16"
target triple = "s390x-ibm-linux"

define signext i32 @dfg_parse() nounwind {
entry:
	br i1 undef, label %if.then2208, label %if.else2360

if.then2208:		; preds = %entry
	br i1 undef, label %bb.nph3189, label %for.end2270

bb.nph3189:		; preds = %if.then2208
	unreachable

for.end2270:		; preds = %if.then2208
	%call2279 = call i64 @strlen(i8* undef) nounwind		; <i64> [#uses=1]
	%add2281 = add i64 0, %call2279		; <i64> [#uses=1]
	%tmp2283 = trunc i64 %add2281 to i32		; <i32> [#uses=1]
	%tmp2284 = alloca i8, i32 %tmp2283, align 2		; <i8*> [#uses=1]
	%yyd.0.i2561.13 = getelementptr i8* %tmp2284, i64 13		; <i8*> [#uses=1]
	store i8 117, i8* %yyd.0.i2561.13
	br label %while.cond.i2558

while.cond.i2558:		; preds = %while.cond.i2558, %for.end2270
	br label %while.cond.i2558

if.else2360:		; preds = %entry
	unreachable
}

declare i64 @strlen(i8* nocapture) nounwind readonly
