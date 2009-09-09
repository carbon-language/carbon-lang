; RUN: llc < %s -march=arm -mattr=+neon

define <16 x i8> @vcombine8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
	ret <16 x i8> %tmp3
}

define <8 x i16> @vcombine16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = shufflevector <4 x i16> %tmp1, <4 x i16> %tmp2, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
	ret <8 x i16> %tmp3
}

define <4 x i32> @vcombine32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = shufflevector <2 x i32> %tmp1, <2 x i32> %tmp2, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
	ret <4 x i32> %tmp3
}

define <4 x float> @vcombinefloat(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = shufflevector <2 x float> %tmp1, <2 x float> %tmp2, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
	ret <4 x float> %tmp3
}

define <2 x i64> @vcombine64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = shufflevector <1 x i64> %tmp1, <1 x i64> %tmp2, <2 x i32> <i32 0, i32 1>
	ret <2 x i64> %tmp3
}
