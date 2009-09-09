; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vaddi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vaddi8:
;CHECK: vadd.i8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = add <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vaddi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vaddi16:
;CHECK: vadd.i16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = add <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vaddi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vaddi32:
;CHECK: vadd.i32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = add <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <1 x i64> @vaddi64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: vaddi64:
;CHECK: vadd.i64
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = add <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

define <2 x float> @vaddf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK: vaddf32:
;CHECK: vadd.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = add <2 x float> %tmp1, %tmp2
	ret <2 x float> %tmp3
}

define <16 x i8> @vaddQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vaddQi8:
;CHECK: vadd.i8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = add <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vaddQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vaddQi16:
;CHECK: vadd.i16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = add <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vaddQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vaddQi32:
;CHECK: vadd.i32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = add <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <2 x i64> @vaddQi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: vaddQi64:
;CHECK: vadd.i64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = add <2 x i64> %tmp1, %tmp2
	ret <2 x i64> %tmp3
}

define <4 x float> @vaddQf32(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK: vaddQf32:
;CHECK: vadd.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = add <4 x float> %tmp1, %tmp2
	ret <4 x float> %tmp3
}
