; RUN: opt < %s -scalarrepl
; PR4286

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "x86_64-undermydesk-freebsd8.0"
	%struct.singlebool = type <{ i8 }>

define zeroext i8 @doit() nounwind {
entry:
	%a = alloca %struct.singlebool, align 1		; <%struct.singlebool*> [#uses=2]
	%storetmp.i = bitcast %struct.singlebool* %a to i1*		; <i1*> [#uses=1]
	store i1 true, i1* %storetmp.i
	%tmp = getelementptr %struct.singlebool* %a, i64 0, i32 0		; <i8*> [#uses=1]
	%tmp1 = load i8* %tmp		; <i8> [#uses=1]
	ret i8 %tmp1
}

