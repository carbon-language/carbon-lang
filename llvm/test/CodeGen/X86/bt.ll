; RUN: llvm-as < %s | llc | grep btl
; RUN: llvm-as < %s | llc -mcpu=pentium4 | grep btl | not grep esp
; RUN: llvm-as < %s | llc -mcpu=penryn   | grep btl | not grep esp
; PR3253

; The register+memory form of the BT instruction should be usable on
; pentium4, however it is currently disabled due to the register+memory
; form having different semantics than the register+register form.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"

define void @test2(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = lshr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, 1		; <i32> [#uses=1]
	%tmp4 = icmp eq i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

declare void @foo()
