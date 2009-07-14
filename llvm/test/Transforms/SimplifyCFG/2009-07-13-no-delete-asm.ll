; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | grep xor
; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"

define void @bar() nounwind {
entry:
	store i16 undef, i16* null
	%asmtmp = call i32 asm sideeffect "xor $0, $0", "=={bx},rm,~{dirflag},~{fpsr},~{flags},~{memory}"(i16 undef) nounwind		; <i32> [#uses=0]
	ret void
}
