; RUN: llc < %s -march=arm -mattr=+neon > %t
; RUN: grep {vmla\\.i8} %t | count 2
; RUN: grep {vmla\\.i16} %t | count 2
; RUN: grep {vmla\\.i32} %t | count 2
; RUN: grep {vmla\\.f32} %t | count 2

define <8 x i8> @vmlai8(<8 x i8>* %A, <8 x i8>* %B, <8 x i8> * %C) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
	%tmp4 = mul <8 x i8> %tmp2, %tmp3
	%tmp5 = add <8 x i8> %tmp1, %tmp4
	ret <8 x i8> %tmp5
}

define <4 x i16> @vmlai16(<4 x i16>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
	%tmp4 = mul <4 x i16> %tmp2, %tmp3
	%tmp5 = add <4 x i16> %tmp1, %tmp4
	ret <4 x i16> %tmp5
}

define <2 x i32> @vmlai32(<2 x i32>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
	%tmp4 = mul <2 x i32> %tmp2, %tmp3
	%tmp5 = add <2 x i32> %tmp1, %tmp4
	ret <2 x i32> %tmp5
}

define <2 x float> @vmlaf32(<2 x float>* %A, <2 x float>* %B, <2 x float>* %C) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = load <2 x float>* %C
	%tmp4 = mul <2 x float> %tmp2, %tmp3
	%tmp5 = add <2 x float> %tmp1, %tmp4
	ret <2 x float> %tmp5
}

define <16 x i8> @vmlaQi8(<16 x i8>* %A, <16 x i8>* %B, <16 x i8> * %C) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = load <16 x i8>* %C
	%tmp4 = mul <16 x i8> %tmp2, %tmp3
	%tmp5 = add <16 x i8> %tmp1, %tmp4
	ret <16 x i8> %tmp5
}

define <8 x i16> @vmlaQi16(<8 x i16>* %A, <8 x i16>* %B, <8 x i16>* %C) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = load <8 x i16>* %C
	%tmp4 = mul <8 x i16> %tmp2, %tmp3
	%tmp5 = add <8 x i16> %tmp1, %tmp4
	ret <8 x i16> %tmp5
}

define <4 x i32> @vmlaQi32(<4 x i32>* %A, <4 x i32>* %B, <4 x i32>* %C) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = load <4 x i32>* %C
	%tmp4 = mul <4 x i32> %tmp2, %tmp3
	%tmp5 = add <4 x i32> %tmp1, %tmp4
	ret <4 x i32> %tmp5
}

define <4 x float> @vmlaQf32(<4 x float>* %A, <4 x float>* %B, <4 x float>* %C) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = load <4 x float>* %C
	%tmp4 = mul <4 x float> %tmp2, %tmp3
	%tmp5 = add <4 x float> %tmp1, %tmp4
	ret <4 x float> %tmp5
}
