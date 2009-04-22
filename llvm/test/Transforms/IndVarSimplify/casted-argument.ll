; RUN: llvm-as < %s | opt -indvars -disable-output
; PR4009

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"

define void @safe_bcopy(i8* %to) nounwind {
entry:
	%cmp11 = icmp ult i8* %to, null		; <i1> [#uses=1]
	br i1 %cmp11, label %loop, label %return

return:		; preds = %entry
	ret void

loop:		; preds = %loop, %if.else
	%pn = phi i8* [ %ge, %loop ], [ null, %entry ]		; <i8*> [#uses=1]
	%cp = ptrtoint i8* %to to i32		; <i32> [#uses=1]
	%su = sub i32 0, %cp		; <i32> [#uses=1]
	%ge = getelementptr i8* %pn, i32 %su		; <i8*> [#uses=2]
	tail call void @bcopy(i8* %ge) nounwind
	br label %loop
}

declare void @bcopy(i8* nocapture) nounwind
