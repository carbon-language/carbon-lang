; RUN: opt < %s -instcombine -S | grep {GLOBAL.*align 16}
; RUN: opt < %s -instcombine -S | grep {tmp = load}
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

@GLOBAL = internal global [4 x i32] zeroinitializer

define <16 x i8> @foo(<2 x i64> %x) {
entry:
	%tmp = load <16 x i8>* bitcast ([4 x i32]* @GLOBAL to <16 x i8>*), align 1
	ret <16 x i8> %tmp
}

