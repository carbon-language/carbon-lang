; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

; XFAIL: *

; CHECK: vmla_8xi8
define <8 x i8> @vmla_8xi8(<8 x i8>* %A, <8 x i8>* %B, <8 x i8> * %C) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
; CHECK: vmla.i8	d16, d18, d17           @ encoding: [0xa1,0x09,0x42,0xf2]
	%tmp4 = mul <8 x i8> %tmp2, %tmp3
	%tmp5 = add <8 x i8> %tmp1, %tmp4
	ret <8 x i8> %tmp5
}

; CHECK: vmla_4xi16
define <4 x i16> @vmla_4xi16(<4 x i16>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
; CHECK: vmla.i16	d16, d18, d17   @ encoding: [0xa1,0x09,0x52,0xf2]
	%tmp4 = mul <4 x i16> %tmp2, %tmp3
	%tmp5 = add <4 x i16> %tmp1, %tmp4
	ret <4 x i16> %tmp5
}

; CHECK: vmla_2xi32
define <2 x i32> @vmla_2xi32(<2 x i32>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
; CHECK: vmla.i32	d16, d18, d17   @ encoding: [0xa1,0x09,0x62,0xf2]
	%tmp4 = mul <2 x i32> %tmp2, %tmp3
	%tmp5 = add <2 x i32> %tmp1, %tmp4
	ret <2 x i32> %tmp5
}

; CHECK: vmla_2xfloat
define <2 x float> @vmla_2xfloat(<2 x float>* %A, <2 x float>* %B, <2 x float>* %C) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = load <2 x float>* %C
; CHECK: vmla.f32	d16, d18, d17   @ encoding: [0xb1,0x0d,0x42,0xf2]
	%tmp4 = fmul <2 x float> %tmp2, %tmp3
	%tmp5 = fadd <2 x float> %tmp1, %tmp4
	ret <2 x float> %tmp5
}

; CHECK: vmla_16xi8
define <16 x i8> @vmla_16xi8(<16 x i8>* %A, <16 x i8>* %B, <16 x i8> * %C) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = load <16 x i8>* %C
; CHECK: vmla.i8	q9, q8, q10             @ encoding: [0xe4,0x29,0x40,0xf2]
	%tmp4 = mul <16 x i8> %tmp2, %tmp3
	%tmp5 = add <16 x i8> %tmp1, %tmp4
	ret <16 x i8> %tmp5
}

; CHECK: vmla_8xi16
define <8 x i16> @vmla_8xi16(<8 x i16>* %A, <8 x i16>* %B, <8 x i16>* %C) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = load <8 x i16>* %C
; CHECK: vmla.i16	q9, q8, q10     @ encoding: [0xe4,0x29,0x50,0xf2]
	%tmp4 = mul <8 x i16> %tmp2, %tmp3
	%tmp5 = add <8 x i16> %tmp1, %tmp4
	ret <8 x i16> %tmp5
}

; CHECK: vmla_4xi32
define <4 x i32> @vmla_4xi32(<4 x i32>* %A, <4 x i32>* %B, <4 x i32>* %C) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = load <4 x i32>* %C
; CHECK: vmla.i32	q9, q8, q10     @ encoding: [0xe4,0x29,0x60,0xf2]
	%tmp4 = mul <4 x i32> %tmp2, %tmp3
	%tmp5 = add <4 x i32> %tmp1, %tmp4
	ret <4 x i32> %tmp5
}

; CHECK: vmla_4xfloat
define <4 x float> @vmla_4xfloat(<4 x float>* %A, <4 x float>* %B, <4 x float>* %C) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = load <4 x float>* %C
; CHECK: vmla.f32	q9, q8, q10     @ encoding: [0xf4,0x2d,0x40,0xf2]
	%tmp4 = fmul <4 x float> %tmp2, %tmp3
	%tmp5 = fadd <4 x float> %tmp1, %tmp4
	ret <4 x float> %tmp5
}

; CHECK: vmlals_8xi8
define <8 x i16> @vmlals_8xi8(<8 x i16>* %A, <8 x i8>* %B, <8 x i8>* %C) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
	%tmp4 = sext <8 x i8> %tmp2 to <8 x i16>
	%tmp5 = sext <8 x i8> %tmp3 to <8 x i16>
; CHECK: vmlal.s8	q8, d19, d18    @ encoding: [0xa2,0x08,0xc3,0xf2]
	%tmp6 = mul <8 x i16> %tmp4, %tmp5
	%tmp7 = add <8 x i16> %tmp1, %tmp6
	ret <8 x i16> %tmp7
}

; CHECK: vmlals_4xi16
define <4 x i32> @vmlals_4xi16(<4 x i32>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
	%tmp4 = sext <4 x i16> %tmp2 to <4 x i32>
	%tmp5 = sext <4 x i16> %tmp3 to <4 x i32>
; CHECK: vmlal.s16	q8, d19, d18    @ encoding: [0xa2,0x08,0xd3,0xf2]
	%tmp6 = mul <4 x i32> %tmp4, %tmp5
	%tmp7 = add <4 x i32> %tmp1, %tmp6
	ret <4 x i32> %tmp7
}

; CHECK: vmlals_2xi32
define <2 x i64> @vmlals_2xi32(<2 x i64>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
	%tmp4 = sext <2 x i32> %tmp2 to <2 x i64>
	%tmp5 = sext <2 x i32> %tmp3 to <2 x i64>
; CHECK: vmlal.s32	q8, d19, d18    @ encoding: [0xa2,0x08,0xe3,0xf2]
	%tmp6 = mul <2 x i64> %tmp4, %tmp5
	%tmp7 = add <2 x i64> %tmp1, %tmp6
	ret <2 x i64> %tmp7
}

; CHECK: vmlalu_8xi8
define <8 x i16> @vmlalu_8xi8(<8 x i16>* %A, <8 x i8>* %B, <8 x i8>* %C) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
	%tmp4 = zext <8 x i8> %tmp2 to <8 x i16>
	%tmp5 = zext <8 x i8> %tmp3 to <8 x i16>
; CHECK: vmlal.u8	q8, d19, d18    @ encoding: [0xa2,0x08,0xc3,0xf3]
	%tmp6 = mul <8 x i16> %tmp4, %tmp5
	%tmp7 = add <8 x i16> %tmp1, %tmp6
	ret <8 x i16> %tmp7
}

; CHECK: vmlalu_4xi16
define <4 x i32> @vmlalu_4xi16(<4 x i32>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
	%tmp4 = zext <4 x i16> %tmp2 to <4 x i32>
	%tmp5 = zext <4 x i16> %tmp3 to <4 x i32>
; CHECK: vmlal.u16	q8, d19, d18    @ encoding: [0xa2,0x08,0xd3,0xf3]
	%tmp6 = mul <4 x i32> %tmp4, %tmp5
	%tmp7 = add <4 x i32> %tmp1, %tmp6
	ret <4 x i32> %tmp7
}

; CHECK: vmlalu_2xi32
define <2 x i64> @vmlalu_2xi32(<2 x i64>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
	%tmp4 = zext <2 x i32> %tmp2 to <2 x i64>
	%tmp5 = zext <2 x i32> %tmp3 to <2 x i64>
; CHECK: vmlal.u32	q8, d19, d18    @ encoding: [0xa2,0x08,0xe3,0xf3]
	%tmp6 = mul <2 x i64> %tmp4, %tmp5
	%tmp7 = add <2 x i64> %tmp1, %tmp6
	ret <2 x i64> %tmp7
}

declare <4 x i32>  @llvm.arm.neon.vqdmlal.v4i32(<4 x i32>, <4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i64>  @llvm.arm.neon.vqdmlal.v2i64(<2 x i64>, <2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vqdmlal_4xi16
define <4 x i32> @vqdmlal_4xi16(<4 x i32>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
; CHECK: vqdmlal.s16	q8, d19, d18    @ encoding: [0xa2,0x09,0xd3,0xf2
	%tmp4 = call <4 x i32> @llvm.arm.neon.vqdmlal.v4i32(<4 x i32> %tmp1, <4 x i16> %tmp2, <4 x i16> %tmp3)
	ret <4 x i32> %tmp4
}

; CHECK: vqdmlal_2xi32
define <2 x i64> @vqdmlal_2xi32(<2 x i64>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
; CHECK: vqdmlal.s32	q8, d19, d18    @ encoding: [0xa2,0x09,0xe3,0xf2]
	%tmp4 = call <2 x i64> @llvm.arm.neon.vqdmlal.v2i64(<2 x i64> %tmp1, <2 x i32> %tmp2, <2 x i32> %tmp3)
	ret <2 x i64> %tmp4
}

; CHECK: vmls_8xi8
define <8 x i8> @vmls_8xi8(<8 x i8>* %A, <8 x i8>* %B, <8 x i8> * %C) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
; CHECK: vmls.i8	d16, d18, d17           @ encoding: [0xa1,0x09,0x42,0xf3]
	%tmp4 = mul <8 x i8> %tmp2, %tmp3
	%tmp5 = sub <8 x i8> %tmp1, %tmp4
	ret <8 x i8> %tmp5
}

; CHECK: vmls_4xi16
define <4 x i16> @vmls_4xi16(<4 x i16>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
; CHECK: vmls.i16	d16, d18, d17   @ encoding: [0xa1,0x09,0x52,0xf3]
	%tmp4 = mul <4 x i16> %tmp2, %tmp3
	%tmp5 = sub <4 x i16> %tmp1, %tmp4
	ret <4 x i16> %tmp5
}

; CHECK: vmls_2xi32
define <2 x i32> @vmls_2xi32(<2 x i32>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
; CHECK: vmls.i32	d16, d18, d17   @ encoding: [0xa1,0x09,0x62,0xf3]
	%tmp4 = mul <2 x i32> %tmp2, %tmp3
	%tmp5 = sub <2 x i32> %tmp1, %tmp4
	ret <2 x i32> %tmp5
}

; CHECK: vmls_2xfloat
define <2 x float> @vmls_2xfloat(<2 x float>* %A, <2 x float>* %B, <2 x float>* %C) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = load <2 x float>* %C
; CHECK: vmls.f32	d16, d18, d17   @ encoding: [0xb1,0x0d,0x62,0xf2]
	%tmp4 = fmul <2 x float> %tmp2, %tmp3
	%tmp5 = fsub <2 x float> %tmp1, %tmp4
	ret <2 x float> %tmp5
}

; CHECK: vmls_16xi8
define <16 x i8> @vmls_16xi8(<16 x i8>* %A, <16 x i8>* %B, <16 x i8> * %C) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = load <16 x i8>* %C
; CHECK: vmls.i8	q9, q8, q10             @ encoding: [0xe4,0x29,0x40,0xf3]
	%tmp4 = mul <16 x i8> %tmp2, %tmp3
	%tmp5 = sub <16 x i8> %tmp1, %tmp4
	ret <16 x i8> %tmp5
}

; CHECK: vmls_8xi16
define <8 x i16> @vmls_8xi16(<8 x i16>* %A, <8 x i16>* %B, <8 x i16>* %C) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = load <8 x i16>* %C
; CHECK: vmls.i16	q9, q8, q10     @ encoding: [0xe4,0x29,0x50,0xf3]
	%tmp4 = mul <8 x i16> %tmp2, %tmp3
	%tmp5 = sub <8 x i16> %tmp1, %tmp4
	ret <8 x i16> %tmp5
}

; CHECK: vmls_4xi32
define <4 x i32> @vmls_4xi32(<4 x i32>* %A, <4 x i32>* %B, <4 x i32>* %C) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = load <4 x i32>* %C
; CHECK: vmls.i32	q9, q8, q10     @ encoding: [0xe4,0x29,0x60,0xf3]
	%tmp4 = mul <4 x i32> %tmp2, %tmp3
	%tmp5 = sub <4 x i32> %tmp1, %tmp4
	ret <4 x i32> %tmp5
}

; CHECK: vmls_4xfloat
define <4 x float> @vmls_4xfloat(<4 x float>* %A, <4 x float>* %B, <4 x float>* %C) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = load <4 x float>* %C
; CHECK: vmls.f32	q9, q8, q10     @ encoding: [0xf4,0x2d,0x60,0xf2]
	%tmp4 = fmul <4 x float> %tmp2, %tmp3
	%tmp5 = fsub <4 x float> %tmp1, %tmp4
	ret <4 x float> %tmp5
}

; CHECK: vmlsls_8xi8
define <8 x i16> @vmlsls_8xi8(<8 x i16>* %A, <8 x i8>* %B, <8 x i8>* %C) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
	%tmp4 = sext <8 x i8> %tmp2 to <8 x i16>
	%tmp5 = sext <8 x i8> %tmp3 to <8 x i16>
; CHECK: vmlsl.s8	q8, d19, d18    @ encoding: [0xa2,0x0a,0xc3,0xf2]
	%tmp6 = mul <8 x i16> %tmp4, %tmp5
	%tmp7 = sub <8 x i16> %tmp1, %tmp6
	ret <8 x i16> %tmp7
}

; CHECK: vmlsls_4xi16
define <4 x i32> @vmlsls_4xi16(<4 x i32>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
	%tmp4 = sext <4 x i16> %tmp2 to <4 x i32>
	%tmp5 = sext <4 x i16> %tmp3 to <4 x i32>
; CHECK: vmlsl.s16	q8, d19, d18    @ encoding: [0xa2,0x0a,0xd3,0xf2]
	%tmp6 = mul <4 x i32> %tmp4, %tmp5
	%tmp7 = sub <4 x i32> %tmp1, %tmp6
	ret <4 x i32> %tmp7
}

; CHECK: vmlsls_2xi32
define <2 x i64> @vmlsls_2xi32(<2 x i64>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
	%tmp4 = sext <2 x i32> %tmp2 to <2 x i64>
	%tmp5 = sext <2 x i32> %tmp3 to <2 x i64>
; CHECK: vmlsl.s32	q8, d19, d18    @ encoding: [0xa2,0x0a,0xe3,0xf2]
	%tmp6 = mul <2 x i64> %tmp4, %tmp5
	%tmp7 = sub <2 x i64> %tmp1, %tmp6
	ret <2 x i64> %tmp7
}

; CHECK: vmlslu_8xi8
define <8 x i16> @vmlslu_8xi8(<8 x i16>* %A, <8 x i8>* %B, <8 x i8>* %C) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
	%tmp4 = zext <8 x i8> %tmp2 to <8 x i16>
	%tmp5 = zext <8 x i8> %tmp3 to <8 x i16>
; CHECK: vmlsl.u8	q8, d19, d18    @ encoding: [0xa2,0x0a,0xc3,0xf3]
	%tmp6 = mul <8 x i16> %tmp4, %tmp5
	%tmp7 = sub <8 x i16> %tmp1, %tmp6
	ret <8 x i16> %tmp7
}

; CHECK: vmlslu_4xi16
define <4 x i32> @vmlslu_4xi16(<4 x i32>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
	%tmp4 = zext <4 x i16> %tmp2 to <4 x i32>
; CHECK: vmlsl.u16	q8, d19, d18    @ encoding: [0xa2,0x0a,0xd3,0xf3]
	%tmp5 = zext <4 x i16> %tmp3 to <4 x i32>
	%tmp6 = mul <4 x i32> %tmp4, %tmp5
	%tmp7 = sub <4 x i32> %tmp1, %tmp6
	ret <4 x i32> %tmp7
}

; CHECK: vmlslu_2xi32
define <2 x i64> @vmlslu_2xi32(<2 x i64>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
	%tmp4 = zext <2 x i32> %tmp2 to <2 x i64>
	%tmp5 = zext <2 x i32> %tmp3 to <2 x i64>
; CHECK: vmlsl.u32	q8, d19, d18    @ encoding: [0xa2,0x0a,0xe3,0xf3]
	%tmp6 = mul <2 x i64> %tmp4, %tmp5
	%tmp7 = sub <2 x i64> %tmp1, %tmp6
	ret <2 x i64> %tmp7
}

declare <4 x i32>  @llvm.arm.neon.vqdmlsl.v4i32(<4 x i32>, <4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i64>  @llvm.arm.neon.vqdmlsl.v2i64(<2 x i64>, <2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vqdmlsl_4xi16
define <4 x i32> @vqdmlsl_4xi16(<4 x i32>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
; CHECK: vqdmlsl.s16	q8, d19, d18    @ encoding: [0xa2,0x0b,0xd3,0xf2]
	%tmp4 = call <4 x i32> @llvm.arm.neon.vqdmlsl.v4i32(<4 x i32> %tmp1, <4 x i16> %tmp2, <4 x i16> %tmp3)
	ret <4 x i32> %tmp4
}

; CHECK: vqdmlsl_2xi32
define <2 x i64> @vqdmlsl_2xi32(<2 x i64>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
; CHECK: vqdmlsl.s32	q8, d19, d18    @ encoding: [0xa2,0x0b,0xe3,0xf2]
	%tmp4 = call <2 x i64> @llvm.arm.neon.vqdmlsl.v2i64(<2 x i64> %tmp1, <2 x i32> %tmp2, <2 x i32> %tmp3)
	ret <2 x i64> %tmp4
}
