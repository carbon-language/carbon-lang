; RUN: llvm-as < %s | llc -march=arm -mattr=+neon > %t
; RUN: grep {vcgt\\.s8} %t | count 2
; RUN: grep {vcgt\\.s16} %t | count 2
; RUN: grep {vcgt\\.s32} %t | count 2
; RUN: grep {vcgt\\.u8} %t | count 2
; RUN: grep {vcgt\\.u16} %t | count 2
; RUN: grep {vcgt\\.u32} %t | count 2
; RUN: grep {vcgt\\.f32} %t | count 2

define <8 x i8> @vcgts8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = vicmp sgt <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vcgts16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = vicmp sgt <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vcgts32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = vicmp sgt <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <8 x i8> @vcgtu8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = vicmp ugt <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vcgtu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = vicmp ugt <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vcgtu32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = vicmp ugt <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <2 x i32> @vcgtf32(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = vfcmp ogt <2 x float> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <16 x i8> @vcgtQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = vicmp sgt <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vcgtQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = vicmp sgt <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vcgtQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = vicmp sgt <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <16 x i8> @vcgtQu8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = vicmp ugt <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vcgtQu16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = vicmp ugt <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vcgtQu32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = vicmp ugt <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <4 x i32> @vcgtQf32(<4 x float>* %A, <4 x float>* %B) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = vfcmp ogt <4 x float> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}
