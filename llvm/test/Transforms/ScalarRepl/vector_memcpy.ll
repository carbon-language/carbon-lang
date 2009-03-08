; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | grep {ret <16 x float> %A}
define <16 x float> @foo(<16 x float> %A) nounwind {
	%tmp = alloca <16 x float>, align 16
	%tmp2 = alloca <16 x float>, align 16
	store <16 x float> %A, <16 x float>* %tmp
	%s = bitcast <16 x float>* %tmp to i8*
	%s2 = bitcast <16 x float>* %tmp2 to i8*
	call void @llvm.memcpy.i64(i8* %s2, i8* %s, i64 64, i32 16)
	
	%R = load <16 x float>* %tmp2
	ret <16 x float> %R
}

declare void @llvm.memcpy.i64(i8* nocapture, i8* nocapture, i64, i32) nounwind

