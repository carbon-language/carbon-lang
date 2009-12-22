; RUN: llc < %s
; PR5703
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-unknown-linux-gnu"

define msp430_intrcc void @foo() nounwind {
entry:
	%fa = call i16* @llvm.frameaddress(i32 0)
	store i16 0, i16* %fa
	ret void
}

declare i16* @llvm.frameaddress(i32)
