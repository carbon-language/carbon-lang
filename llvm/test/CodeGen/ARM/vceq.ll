; RUN: llvm-as < %s | llc -march=arm -mattr=+neon > %t
; RUN: grep {vceq\\.i8} %t | count 2
; RUN: grep {vceq\\.i16} %t | count 2
; RUN: grep {vceq\\.i32} %t | count 2
; RUN: grep {vceq\\.f32} %t | count 2

define <8 x i8> @vceqi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = vicmp eq <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vceqi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = vicmp eq <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vceqi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = vicmp eq <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <2 x i32> @vceqf32(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = vfcmp oeq <2 x float> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <16 x i8> @vceqQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = vicmp eq <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vceqQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = vicmp eq <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vceqQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = vicmp eq <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <4 x i32> @vceqQf32(<4 x float>* %A, <4 x float>* %B) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = vfcmp oeq <4 x float> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}
