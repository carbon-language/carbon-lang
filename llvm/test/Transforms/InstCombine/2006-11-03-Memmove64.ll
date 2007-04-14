; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:    not grep memmove.i32
; Instcombine was trying to turn this into a memmove.i32

target datalayout = "e-p:64:64"
target endian = little
target pointersize = 64
target triple = "alphaev67-unknown-linux-gnu"
%str10 = internal constant [1 x sbyte] zeroinitializer		; <[1 x sbyte]*> [#uses=1]

implementation   ; Functions:

void %do_join(sbyte* %b) {
entry:
	call void %llvm.memmove.i64( sbyte* %b, sbyte* getelementptr ([1 x sbyte]* %str10, int 0, ulong 0), ulong 1, uint 1 )
	ret void
}

declare void %llvm.memmove.i64(sbyte*, sbyte*, ulong, uint)
