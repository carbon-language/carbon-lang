; RUN: llvm-as < %s | llc
; PR4668
; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"

define i32 @x(i32 %qscale) nounwind {
entry:
	%temp_block = alloca [64 x i16], align 16		; <[64 x i16]*> [#uses=0]
	%tmp = call i32 asm sideeffect "xor %edx, %edx", "={dx},~{dirflag},~{fpsr},~{flags}"() nounwind		; <i32> [#uses=1]
	br i1 undef, label %if.end78, label %if.then28

if.then28:		; preds = %entry
	br label %if.end78

if.end78:		; preds = %if.then28, %entry
	%level.1 = phi i32 [ %tmp, %if.then28 ], [ 0, %entry ]		; <i32> [#uses=1]
	%add.ptr1 = getelementptr [64 x i16]* null, i32 0, i32 %qscale		; <i16*> [#uses=1]
	%add.ptr2 = getelementptr [64 x i16]* null, i32 1, i32 %qscale		; <i16*> [#uses=1]
	%add.ptr3 = getelementptr [64 x i16]* null, i32 2, i32 %qscale		; <i16*> [#uses=1]
	%add.ptr4 = getelementptr [64 x i16]* null, i32 3, i32 %qscale		; <i16*> [#uses=1]
	%add.ptr5 = getelementptr [64 x i16]* null, i32 4, i32 %qscale		; <i16*> [#uses=1]
	%add.ptr6 = getelementptr [64 x i16]* null, i32 5, i32 %qscale		; <i16*> [#uses=1]
	%tmp1 = call i32 asm sideeffect "nop", "={ax},r,r,r,r,r,0,~{dirflag},~{fpsr},~{flags}"(i16* %add.ptr6, i16* %add.ptr5, i16* %add.ptr4, i16* %add.ptr3, i16* %add.ptr2, i16* %add.ptr1) nounwind		; <i32> [#uses=0]
	ret i32 %level.1
}
