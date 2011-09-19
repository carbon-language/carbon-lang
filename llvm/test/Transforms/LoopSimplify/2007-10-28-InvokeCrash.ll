; RUN: llvm-as < %s | opt -loop-simplify -disable-output
; PR1752
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-s0:0:64-f80:32:32"
target triple = "i686-pc-mingw32"

define void @func() {
bb_init:
	br label %bb_main

bb_main:
        br label %invcont17.normaldest

invcont17.normaldest917:		; No predecessors!
	%tmp23 = invoke i32 @foo()
			to label %invcont17.normaldest unwind label %invcont17.normaldest.normaldest

invcont17.normaldest:		; preds = %invcont17.normaldest917, %bb_main
	br label %bb_main

invcont17.normaldest.normaldest:		; No predecessors!
        %eh_ptr = call i8* @llvm.eh.exception()
	%eh_select = tail call i32 (i8*, i8*, ...)* @llvm.eh.selector(i8* %eh_ptr, i8* bitcast (void ()* @__gxx_personality_v0 to i8*), i8* null)
        store i32 %tmp23, i32* undef
	br label %bb_main
}

declare i32 @foo()

declare i8* @llvm.eh.exception()

declare i32 @llvm.eh.selector(i8*, i8*, ...)

declare void @__gxx_personality_v0()
