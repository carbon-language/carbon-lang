; RUN: llvm-as < %s | llc -march=arm -mattr=+neon > %t
; RUN: grep {vcge\\.s8} %t | count 2
; RUN: grep {vcge\\.s16} %t | count 2
; RUN: grep {vcge\\.s32} %t | count 2
; RUN: grep {vcge\\.u8} %t | count 2
; RUN: grep {vcge\\.u16} %t | count 2
; RUN: grep {vcge\\.u32} %t | count 2
; RUN: grep {vcge\\.f32} %t | count 2

define <8 x i8> @vcges8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = vicmp sge <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vcges16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = vicmp sge <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vcges32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = vicmp sge <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <8 x i8> @vcgeu8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = vicmp uge <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vcgeu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = vicmp uge <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vcgeu32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = vicmp uge <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <2 x i32> @vcgef32(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = vfcmp oge <2 x float> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <16 x i8> @vcgeQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = vicmp sge <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vcgeQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = vicmp sge <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vcgeQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = vicmp sge <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <16 x i8> @vcgeQu8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = vicmp uge <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vcgeQu16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = vicmp uge <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vcgeQu32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = vicmp uge <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <4 x i32> @vcgeQf32(<4 x float>* %A, <4 x float>* %B) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = vfcmp oge <4 x float> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}
