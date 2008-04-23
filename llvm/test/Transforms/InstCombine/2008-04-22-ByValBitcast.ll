;; The bitcast cannot be eliminated because byval arguments need
;; the correct type, or at least a type of the correct size.
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep bitcast
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9"
	%struct.NSRect = type { [4 x float] }

define void @foo(i8* %context) nounwind  {
entry:
	%context_addr = alloca i8*		; <i8**> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i8* %context, i8** %context_addr
	%tmp = load i8** %context_addr, align 4		; <i8*> [#uses=1]
	%tmp1 = bitcast i8* %tmp to %struct.NSRect*		; <%struct.NSRect*> [#uses=1]
	call void (i32, ...)* @bar( i32 3, %struct.NSRect* byval align 4  %tmp1 ) nounwind 
	br label %return
return:		; preds = %entry
	ret void
}

declare void @bar(i32, ...)
