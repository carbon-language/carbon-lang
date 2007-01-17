; RUN: llvm-upgrade < %s | llvm-as | llc
; PR1029

target datalayout = "e-p:64:64"
target endian = little
target pointersize = 64
target triple = "x86_64-unknown-linux-gnu"

implementation   ; Functions:

void %frame_dummy() {
entry:
	%tmp1 = tail call void (sbyte*)* (void (sbyte*)*)* asm "", "=r,0,~{dirflag},~{fpsr},~{flags}"( void (sbyte*)* null )		; <void (sbyte*)*> [#uses=0]
	ret void
}
