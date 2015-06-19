; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

define <8 x i8> @vuzpi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
; CHECK-LABEL: vuzpi8:
; CHECK:       @ BB#0:
; CHECK-NEXT:    vldr d16, [r1]
; CHECK-NEXT:    vldr d17, [r0]
; CHECK-NEXT:    vuzp.8 d17, d16
; CHECK-NEXT:    vadd.i8 d16, d17, d16
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
	%tmp4 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
        %tmp5 = add <8 x i8> %tmp3, %tmp4
	ret <8 x i8> %tmp5
}

define <16 x i8> @vuzpi8_Qres(<8 x i8>* %A, <8 x i8>* %B) nounwind {
; CHECK-LABEL: vuzpi8_Qres:
; CHECK:       @ BB#0:
; CHECK-NEXT:    vldr d17, [r1]
; CHECK-NEXT:    vldr d16, [r0]
; CHECK-NEXT:    vuzp.8 d16, d17
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
	ret <16 x i8> %tmp3
}

define <4 x i16> @vuzpi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
; CHECK-LABEL: vuzpi16:
; CHECK:       @ BB#0:
; CHECK-NEXT:    vldr d16, [r1]
; CHECK-NEXT:    vldr d17, [r0]
; CHECK-NEXT:    vuzp.16 d17, d16
; CHECK-NEXT:    vadd.i16 d16, d17, d16
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = shufflevector <4 x i16> %tmp1, <4 x i16> %tmp2, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
	%tmp4 = shufflevector <4 x i16> %tmp1, <4 x i16> %tmp2, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
        %tmp5 = add <4 x i16> %tmp3, %tmp4
	ret <4 x i16> %tmp5
}

define <8 x i16> @vuzpi16_Qres(<4 x i16>* %A, <4 x i16>* %B) nounwind {
; CHECK-LABEL: vuzpi16_Qres:
; CHECK:       @ BB#0:
; CHECK-NEXT:    vldr d17, [r1]
; CHECK-NEXT:    vldr d16, [r0]
; CHECK-NEXT:    vuzp.16 d16, d17
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = shufflevector <4 x i16> %tmp1, <4 x i16> %tmp2, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 1, i32 3, i32 5, i32 7>
	ret <8 x i16> %tmp3
}

; VUZP.32 is equivalent to VTRN.32 for 64-bit vectors.

define <16 x i8> @vuzpQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
; CHECK-LABEL: vuzpQi8:
; CHECK:       @ BB#0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r1]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r0]
; CHECK-NEXT:    vuzp.8 q9, q8
; CHECK-NEXT:    vadd.i8 q8, q9, q8
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
	%tmp4 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
        %tmp5 = add <16 x i8> %tmp3, %tmp4
	ret <16 x i8> %tmp5
}

define <32 x i8> @vuzpQi8_QQres(<16 x i8>* %A, <16 x i8>* %B) nounwind {
; CHECK-LABEL: vuzpQi8_QQres:
; CHECK:       @ BB#0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r2]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r1]
; CHECK-NEXT:    vuzp.8 q9, q8
; CHECK-NEXT:    vst1.8 {d18, d19}, [r0:128]!
; CHECK-NEXT:    vst1.64 {d16, d17}, [r0:128]
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <32 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30, i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
	ret <32 x i8> %tmp3
}

define <8 x i16> @vuzpQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
; CHECK-LABEL: vuzpQi16:
; CHECK:       @ BB#0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r1]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r0]
; CHECK-NEXT:    vuzp.16 q9, q8
; CHECK-NEXT:    vadd.i16 q8, q9, q8
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
	%tmp4 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
        %tmp5 = add <8 x i16> %tmp3, %tmp4
	ret <8 x i16> %tmp5
}

define <16 x i16> @vuzpQi16_QQres(<8 x i16>* %A, <8 x i16>* %B) nounwind {
; CHECK-LABEL: vuzpQi16_QQres:
; CHECK:       @ BB#0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r2]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r1]
; CHECK-NEXT:    vuzp.16 q9, q8
; CHECK-NEXT:    vst1.16 {d18, d19}, [r0:128]!
; CHECK-NEXT:    vst1.64 {d16, d17}, [r0:128]
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
	ret <16 x i16> %tmp3
}

define <4 x i32> @vuzpQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
; CHECK-LABEL: vuzpQi32:
; CHECK:       @ BB#0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r1]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r0]
; CHECK-NEXT:    vuzp.32 q9, q8
; CHECK-NEXT:    vadd.i32 q8, q9, q8
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = shufflevector <4 x i32> %tmp1, <4 x i32> %tmp2, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
	%tmp4 = shufflevector <4 x i32> %tmp1, <4 x i32> %tmp2, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
        %tmp5 = add <4 x i32> %tmp3, %tmp4
	ret <4 x i32> %tmp5
}

define <8 x i32> @vuzpQi32_QQres(<4 x i32>* %A, <4 x i32>* %B) nounwind {
; CHECK-LABEL: vuzpQi32_QQres:
; CHECK:       @ BB#0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r2]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r1]
; CHECK-NEXT:    vuzp.32 q9, q8
; CHECK-NEXT:    vst1.32 {d18, d19}, [r0:128]!
; CHECK-NEXT:    vst1.64 {d16, d17}, [r0:128]
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = shufflevector <4 x i32> %tmp1, <4 x i32> %tmp2, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 1, i32 3, i32 5, i32 7>
	ret <8 x i32> %tmp3
}

define <4 x float> @vuzpQf(<4 x float>* %A, <4 x float>* %B) nounwind {
; CHECK-LABEL: vuzpQf:
; CHECK:       @ BB#0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r1]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r0]
; CHECK-NEXT:    vuzp.32 q9, q8
; CHECK-NEXT:    vadd.f32 q8, q9, q8
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <4 x float>, <4 x float>* %A
	%tmp2 = load <4 x float>, <4 x float>* %B
	%tmp3 = shufflevector <4 x float> %tmp1, <4 x float> %tmp2, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
	%tmp4 = shufflevector <4 x float> %tmp1, <4 x float> %tmp2, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
        %tmp5 = fadd <4 x float> %tmp3, %tmp4
	ret <4 x float> %tmp5
}

define <8 x float> @vuzpQf_QQres(<4 x float>* %A, <4 x float>* %B) nounwind {
; CHECK-LABEL: vuzpQf_QQres:
; CHECK:       @ BB#0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r2]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r1]
; CHECK-NEXT:    vuzp.32 q9, q8
; CHECK-NEXT:    vst1.32 {d18, d19}, [r0:128]!
; CHECK-NEXT:    vst1.64 {d16, d17}, [r0:128]
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <4 x float>, <4 x float>* %A
	%tmp2 = load <4 x float>, <4 x float>* %B
	%tmp3 = shufflevector <4 x float> %tmp1, <4 x float> %tmp2, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 1, i32 3, i32 5, i32 7>
	ret <8 x float> %tmp3
}

; Undef shuffle indices should not prevent matching to VUZP:

define <8 x i8> @vuzpi8_undef(<8 x i8>* %A, <8 x i8>* %B) nounwind {
; CHECK-LABEL: vuzpi8_undef:
; CHECK:       @ BB#0:
; CHECK-NEXT:    vldr d16, [r1]
; CHECK-NEXT:    vldr d17, [r0]
; CHECK-NEXT:    vuzp.8 d17, d16
; CHECK-NEXT:    vadd.i8 d16, d17, d16
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 0, i32 2, i32 undef, i32 undef, i32 8, i32 10, i32 12, i32 14>
	%tmp4 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 undef, i32 undef, i32 13, i32 15>
        %tmp5 = add <8 x i8> %tmp3, %tmp4
	ret <8 x i8> %tmp5
}

define <16 x i8> @vuzpi8_undef_Qres(<8 x i8>* %A, <8 x i8>* %B) nounwind {
; CHECK-LABEL: vuzpi8_undef_Qres:
; CHECK:       @ BB#0:
; CHECK-NEXT:    vldr d17, [r1]
; CHECK-NEXT:    vldr d16, [r0]
; CHECK-NEXT:    vuzp.8 d16, d17
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <16 x i32> <i32 0, i32 2, i32 undef, i32 undef, i32 8, i32 10, i32 12, i32 14, i32 1, i32 3, i32 5, i32 7, i32 undef, i32 undef, i32 13, i32 15>
	ret <16 x i8> %tmp3
}

define <8 x i16> @vuzpQi16_undef(<8 x i16>* %A, <8 x i16>* %B) nounwind {
; CHECK-LABEL: vuzpQi16_undef:
; CHECK:       @ BB#0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r1]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r0]
; CHECK-NEXT:    vuzp.16 q9, q8
; CHECK-NEXT:    vadd.i16 q8, q9, q8
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> <i32 0, i32 undef, i32 4, i32 undef, i32 8, i32 10, i32 12, i32 14>
	%tmp4 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> <i32 1, i32 3, i32 5, i32 undef, i32 undef, i32 11, i32 13, i32 15>
        %tmp5 = add <8 x i16> %tmp3, %tmp4
	ret <8 x i16> %tmp5
}

define <16 x i16> @vuzpQi16_undef_QQres(<8 x i16>* %A, <8 x i16>* %B) nounwind {
; CHECK-LABEL: vuzpQi16_undef_QQres:
; CHECK:       @ BB#0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r2]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r1]
; CHECK-NEXT:    vuzp.16 q9, q8
; CHECK-NEXT:    vst1.16 {d18, d19}, [r0:128]!
; CHECK-NEXT:    vst1.64 {d16, d17}, [r0:128]
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <16 x i32> <i32 0, i32 undef, i32 4, i32 undef, i32 8, i32 10, i32 12, i32 14, i32 1, i32 3, i32 5, i32 undef, i32 undef, i32 11, i32 13, i32 15>
	ret <16 x i16> %tmp3
}
