; RUN: opt -S -dse < %s | FileCheck %s

declare i8* @llvm.init.trampoline(i8*, i8*, i8*)

declare void @f()

define void @unused_trampoline() {
; CHECK: @unused_trampoline
	%storage = alloca [10 x i8], align 16		; <[10 x i8]*> [#uses=1]
; CHECK-NOT: alloca
	%cast = getelementptr [10 x i8]* %storage, i32 0, i32 0		; <i8*> [#uses=1]
	%tramp = call i8* @llvm.init.trampoline( i8* %cast, i8* bitcast (void ()* @f to i8*), i8* null )		; <i8*> [#uses=1]
; CHECK-NOT: trampoline
	ret void
; CHECK: ret void
}
