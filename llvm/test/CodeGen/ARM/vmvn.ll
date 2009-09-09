; RUN: llc < %s -march=arm -mattr=+neon > %t
; RUN: grep vmvn %t | count 8
; Note: function names do not include "vmvn" to allow simple grep for opcodes

define <8 x i8> @v_mvni8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = xor <8 x i8> %tmp1, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	ret <8 x i8> %tmp2
}

define <4 x i16> @v_mvni16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = xor <4 x i16> %tmp1, < i16 -1, i16 -1, i16 -1, i16 -1 >
	ret <4 x i16> %tmp2
}

define <2 x i32> @v_mvni32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = xor <2 x i32> %tmp1, < i32 -1, i32 -1 >
	ret <2 x i32> %tmp2
}

define <1 x i64> @v_mvni64(<1 x i64>* %A) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = xor <1 x i64> %tmp1, < i64 -1 >
	ret <1 x i64> %tmp2
}

define <16 x i8> @v_mvnQi8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = xor <16 x i8> %tmp1, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	ret <16 x i8> %tmp2
}

define <8 x i16> @v_mvnQi16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = xor <8 x i16> %tmp1, < i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1 >
	ret <8 x i16> %tmp2
}

define <4 x i32> @v_mvnQi32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = xor <4 x i32> %tmp1, < i32 -1, i32 -1, i32 -1, i32 -1 >
	ret <4 x i32> %tmp2
}

define <2 x i64> @v_mvnQi64(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = xor <2 x i64> %tmp1, < i64 -1, i64 -1 >
	ret <2 x i64> %tmp2
}
