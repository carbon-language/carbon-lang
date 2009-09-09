; RUN: llc < %s

target datalayout = "E-p:64:64:64-i1:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128"
target triple = "s390x-ibm-linux"
	%struct.re_pattern_buffer = type <{ i8*, i64, i64, i64, i8*, i8*, i64, i8, i8, i8, i8, i8, i8, i8, i8 }>
	%struct.re_registers = type <{ i32, i8, i8, i8, i8, i32*, i32* }>

define i32 @xre_search_2(%struct.re_pattern_buffer* nocapture %bufp, i8* %string1, i32 %size1, i8* %string2, i32 %size2, i32 %startpos, i32 %range, %struct.re_registers* %regs, i32 %stop) nounwind {
entry:
	%cmp17.i = icmp slt i32 undef, %startpos		; <i1> [#uses=1]
	%or.cond.i = or i1 undef, %cmp17.i		; <i1> [#uses=1]
	br i1 %or.cond.i, label %byte_re_search_2.exit, label %if.then20.i

if.then20.i:		; preds = %entry
	ret i32 -2

byte_re_search_2.exit:		; preds = %entry
	ret i32 -1
}
