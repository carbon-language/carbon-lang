; RUN: llvm-as < %s | opt -gvn | llvm-dis | grep undef
; PR2503
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin9.3.0"
@g_3 = external global i8		; <i8*> [#uses=2]

define i32 @func_1() nounwind  {
entry:
	br i1 false, label %ifelse, label %ifthen

ifthen:		; preds = %entry
	br label %ifend

ifelse:		; preds = %entry
	%tmp3 = load i8* @g_3		; <i8> [#uses=0]
	br label %forcond.thread

forcond.thread:		; preds = %ifelse
	br label %afterfor

forcond:		; preds = %forinc
	br i1 false, label %afterfor, label %forbody

forbody:		; preds = %forcond
	br label %forinc

forinc:		; preds = %forbody
	br label %forcond

afterfor:		; preds = %forcond, %forcond.thread
	%tmp10 = load i8* @g_3		; <i8> [#uses=0]
	br label %ifend

ifend:		; preds = %afterfor, %ifthen
	ret i32 0
}
