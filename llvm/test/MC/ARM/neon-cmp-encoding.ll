; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

; XFAIL: *

; FIXME: We cannot currently test the following instructions, which are 
; currently marked as for-disassembly only in the .td files:
;  - VCEQz
;  - VCGEz, VCLEz
;  - VCGTz, VCLTz

; CHECK: vceq_8xi8
define <8 x i8> @vceq_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vceq.i8	d16, d16, d17           @ encoding: [0xb1,0x08,0x40,0xf3]
	%tmp3 = icmp eq <8 x i8> %tmp1, %tmp2
  %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

; CHECK: vceq_4xi16
define <4 x i16> @vceq_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vceq.i16	d16, d16, d17   @ encoding: [0xb1,0x08,0x50,0xf3]
	%tmp3 = icmp eq <4 x i16> %tmp1, %tmp2
  %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

; CHECK: vceq_2xi32
define <2 x i32> @vceq_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vceq.i32	d16, d16, d17   @ encoding: [0xb1,0x08,0x60,0xf3]
	%tmp3 = icmp eq <2 x i32> %tmp1, %tmp2
  %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

; CHECK: vceq_2xfloat
define <2 x i32> @vceq_2xfloat(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
; CHECK: vceq.f32	d16, d16, d17   @ encoding: [0xa1,0x0e,0x40,0xf2]
	%tmp3 = fcmp oeq <2 x float> %tmp1, %tmp2
  %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

; CHECK: vceq_16xi8
define <16 x i8> @vceq_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vceq.i8	q8, q8, q9              @ encoding: [0xf2,0x08,0x40,0xf3]
	%tmp3 = icmp eq <16 x i8> %tmp1, %tmp2
  %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

; CHECK: vceq_8xi16
define <8 x i16> @vceq_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vceq.i16	q8, q8, q9      @ encoding: [0xf2,0x08,0x50,0xf3]
	%tmp3 = icmp eq <8 x i16> %tmp1, %tmp2
  %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

; CHECK: vceq_4xi32
define <4 x i32> @vceq_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vceq.i32	q8, q8, q9      @ encoding: [0xf2,0x08,0x60,0xf3]
	%tmp3 = icmp eq <4 x i32> %tmp1, %tmp2
  %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

; CHECK: vceq_4xfloat
define <4 x i32> @vceq_4xfloat(<4 x float>* %A, <4 x float>* %B) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
; CHECK: vceq.f32	q8, q8, q9      @ encoding: [0xe2,0x0e,0x40,0xf2]
	%tmp3 = fcmp oeq <4 x float> %tmp1, %tmp2
  %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

; CHECK: vcges_8xi8
define <8 x i8> @vcges_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vcge.s8	d16, d16, d17           @ encoding: [0xb1,0x03,0x40,0xf2]
	%tmp3 = icmp sge <8 x i8> %tmp1, %tmp2
  %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

; CHECK: vcges_4xi16
define <4 x i16> @vcges_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = icmp sge <4 x i16> %tmp1, %tmp2
; CHECK: vcge.s16	d16, d16, d17   @ encoding: [0xb1,0x03,0x50,0xf2]
  %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

; CHECK: vcges_2xi32
define <2 x i32> @vcges_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vcge.s32	d16, d16, d17   @ encoding: [0xb1,0x03,0x60,0xf2]
	%tmp3 = icmp sge <2 x i32> %tmp1, %tmp2
  %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

; CHECK: vcgeu_8xi8
define <8 x i8> @vcgeu_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vcge.u8	d16, d16, d17           @ encoding: [0xb1,0x03,0x40,0xf3]
	%tmp3 = icmp uge <8 x i8> %tmp1, %tmp2
  %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

; CHECK: vcgeu_4xi16
define <4 x i16> @vcgeu_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vcge.u16	d16, d16, d17   @ encoding: [0xb1,0x03,0x50,0xf3]
	%tmp3 = icmp uge <4 x i16> %tmp1, %tmp2
  %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

; CHECK: vcgeu_2xi32
define <2 x i32> @vcgeu_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = icmp uge <2 x i32> %tmp1, %tmp2
; CHECK: vcge.u32	d16, d16, d17   @ encoding: [0xb1,0x03,0x60,0xf3]
  %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

; CHECK: vcge_2xfloat
define <2 x i32> @vcge_2xfloat(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
; CHECK: vcge.f32	d16, d16, d17   @ encoding: [0xa1,0x0e,0x40,0xf3]
	%tmp3 = fcmp oge <2 x float> %tmp1, %tmp2
  %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

; CHECK: vcges_16xi8
define <16 x i8> @vcges_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vcge.s8	q8, q8, q9              @ encoding: [0xf2,0x03,0x40,0xf2]
	%tmp3 = icmp sge <16 x i8> %tmp1, %tmp2
  %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

; CHECK: vcges_8xi16
define <8 x i16> @vcges_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vcge.s16	q8, q8, q9      @ encoding: [0xf2,0x03,0x50,0xf2]
	%tmp3 = icmp sge <8 x i16> %tmp1, %tmp2
  %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

; CHECK: vcges_4xi32
define <4 x i32> @vcges_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vcge.s32	q8, q8, q9      @ encoding: [0xf2,0x03,0x60,0xf2]
	%tmp3 = icmp sge <4 x i32> %tmp1, %tmp2
  %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

; CHECK: vcgeu_16xi8
define <16 x i8> @vcgeu_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vcge.u8	q8, q8, q9              @ encoding: [0xf2,0x03,0x40,0xf3]
	%tmp3 = icmp uge <16 x i8> %tmp1, %tmp2
  %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

; CHECK: vcgeu_8xi16
define <8 x i16> @vcgeu_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vcge.u16	q8, q8, q9      @ encoding: [0xf2,0x03,0x50,0xf3]
	%tmp3 = icmp uge <8 x i16> %tmp1, %tmp2
  %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

; CHECK: vcgeu_4xi32
define <4 x i32> @vcgeu_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vcge.u32	q8, q8, q9      @ encoding: [0xf2,0x03,0x60,0xf3]
	%tmp3 = icmp uge <4 x i32> %tmp1, %tmp2
  %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

; CHECK: vcge_4xfloat
define <4 x i32> @vcge_4xfloat(<4 x float>* %A, <4 x float>* %B) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
; CHECK: vcge.f32	q8, q8, q9      @ encoding: [0xe2,0x0e,0x40,0xf3]
	%tmp3 = fcmp oge <4 x float> %tmp1, %tmp2
  %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

declare <2 x i32> @llvm.arm.neon.vacged(<2 x float>, <2 x float>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vacgeq(<4 x float>, <4 x float>) nounwind readnone

; CHECK: vacge_2xfloat
define <2 x i32> @vacge_2xfloat(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
; vacge.f32	d16, d16, d17   @ encoding: [0xb1,0x0e,0x40,0xf3]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vacged(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x i32> %tmp3
}

; CHECK: vacge_4xfloat
define <4 x i32> @vacge_4xfloat(<4 x float>* %A, <4 x float>* %B) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
; CHECK: vacge.f32	q8, q8, q9      @ encoding: [0xf2,0x0e,0x40,0xf3]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vacgeq(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x i32> %tmp3
}

; CHECK: vcgts_8xi8
define <8 x i8> @vcgts_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vcgt.s8	d16, d16, d17           @ encoding: [0xa1,0x03,0x40,0xf2]
	%tmp3 = icmp sgt <8 x i8> %tmp1, %tmp2
  %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

; CHECK: vcgts_4xi16
define <4 x i16> @vcgts_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vcgt.s16	d16, d16, d17   @ encoding: [0xa1,0x03,0x50,0xf2]
	%tmp3 = icmp sgt <4 x i16> %tmp1, %tmp2
  %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

; CHECK: vcgts_2xi32
define <2 x i32> @vcgts_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vcgt.s32	d16, d16, d17   @ encoding: [0xa1,0x03,0x60,0xf2]
	%tmp3 = icmp sgt <2 x i32> %tmp1, %tmp2
  %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

; CHECK: vcgtu_8xi8
define <8 x i8> @vcgtu_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vcgt.u8	d16, d16, d17           @ encoding: [0xa1,0x03,0x40,0xf3]
	%tmp3 = icmp ugt <8 x i8> %tmp1, %tmp2
  %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

; CHECK: vcgtu_4xi16
define <4 x i16> @vcgtu_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vcgt.u16	d16, d16, d17   @ encoding: [0xa1,0x03,0x50,0xf3]
	%tmp3 = icmp ugt <4 x i16> %tmp1, %tmp2
  %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

; CHECK: vcgtu_2xi32
define <2 x i32> @vcgtu_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vcgt.u32	d16, d16, d17   @ encoding: [0xa1,0x03,0x60,0xf3]
	%tmp3 = icmp ugt <2 x i32> %tmp1, %tmp2
  %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

; CHECK: vcgt_2xfloat
define <2 x i32> @vcgt_2xfloat(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
; CHECK: vcgt.f32	d16, d16, d17   @ encoding: [0xa1,0x0e,0x60,0xf3]
	%tmp3 = fcmp ogt <2 x float> %tmp1, %tmp2
  %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

; CHECK: vcgts_16xi8
define <16 x i8> @vcgts_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vcgt.s8	q8, q8, q9              @ encoding: [0xe2,0x03,0x40,0xf2]
	%tmp3 = icmp sgt <16 x i8> %tmp1, %tmp2
  %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

; CHECK: vcgts_8xi16
define <8 x i16> @vcgts_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vcgt.s16	q8, q8, q9      @ encoding: [0xe2,0x03,0x50,0xf2]
	%tmp3 = icmp sgt <8 x i16> %tmp1, %tmp2
  %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

; CHECK: vcgts_4xi32
define <4 x i32> @vcgts_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vcgt.s32	q8, q8, q9      @ encoding: [0xe2,0x03,0x60,0xf2]
	%tmp3 = icmp sgt <4 x i32> %tmp1, %tmp2
  %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

; CHECK: vcgtu_16xi8
define <16 x i8> @vcgtu_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vcgt.u8	q8, q8, q9              @ encoding: [0xe2,0x03,0x40,0xf3]
	%tmp3 = icmp ugt <16 x i8> %tmp1, %tmp2
  %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

; CHECK: vcgtu_8xi16
define <8 x i16> @vcgtu_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vcgt.u16	q8, q8, q9      @ encoding: [0xe2,0x03,0x50,0xf3]
	%tmp3 = icmp ugt <8 x i16> %tmp1, %tmp2
  %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

; CHECK: vcgtu_4xi32
define <4 x i32> @vcgtu_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vcgt.u32	q8, q8, q9      @ encoding: [0xe2,0x03,0x60,0xf3]
	%tmp3 = icmp ugt <4 x i32> %tmp1, %tmp2
  %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

; CHECK: vcgt_4xfloat
define <4 x i32> @vcgt_4xfloat(<4 x float>* %A, <4 x float>* %B) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
; CHECK: vcgt.f32	q8, q8, q9      @ encoding: [0xe2,0x0e,0x60,0xf3]
	%tmp3 = fcmp ogt <4 x float> %tmp1, %tmp2
  %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

declare <2 x i32> @llvm.arm.neon.vacgtd(<2 x float>, <2 x float>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vacgtq(<4 x float>, <4 x float>) nounwind readnone

; CHECK: vacgt_2xfloat
define <2 x i32> @vacgt_2xfloat(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
; CHECK: vacgt.f32	d16, d16, d17   @ encoding: [0xb1,0x0e,0x60,0xf3]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vacgtd(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x i32> %tmp3
}

; CHECK: vacgt_4xfloat
define <4 x i32> @vacgt_4xfloat(<4 x float>* %A, <4 x float>* %B) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
; CHECK: vacgt.f32	q8, q8, q9      @ encoding: [0xf2,0x0e,0x60,0xf3]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vacgtq(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x i32> %tmp3
}

; CHECK: vtst_8xi8
define <8 x i8> @vtst_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vtst.8	d16, d16, d17           @ encoding: [0xb1,0x08,0x40,0xf2]
	%tmp3 = and <8 x i8> %tmp1, %tmp2
	%tmp4 = icmp ne <8 x i8> %tmp3, zeroinitializer
  %tmp5 = sext <8 x i1> %tmp4 to <8 x i8>
	ret <8 x i8> %tmp5
}

; CHECK: vtst_4xi16
define <4 x i16> @vtst_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vtst.16	d16, d16, d17           @ encoding: [0xb1,0x08,0x50,0xf2]
	%tmp3 = and <4 x i16> %tmp1, %tmp2
	%tmp4 = icmp ne <4 x i16> %tmp3, zeroinitializer
  %tmp5 = sext <4 x i1> %tmp4 to <4 x i16>
	ret <4 x i16> %tmp5
}

; CHECK: vtst_2xi32
define <2 x i32> @vtst_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vtst.32	d16, d16, d17           @ encoding: [0xb1,0x08,0x60,0xf2]
	%tmp3 = and <2 x i32> %tmp1, %tmp2
	%tmp4 = icmp ne <2 x i32> %tmp3, zeroinitializer
  %tmp5 = sext <2 x i1> %tmp4 to <2 x i32>
	ret <2 x i32> %tmp5
}

; CHECK: vtst_16xi8
define <16 x i8> @vtst_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vtst.8	q8, q8, q9              @ encoding: [0xf2,0x08,0x40,0xf2]
	%tmp3 = and <16 x i8> %tmp1, %tmp2
	%tmp4 = icmp ne <16 x i8> %tmp3, zeroinitializer
  %tmp5 = sext <16 x i1> %tmp4 to <16 x i8>
	ret <16 x i8> %tmp5
}

; CHECK: vtst_8xi16
define <8 x i16> @vtst_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vtst.16	q8, q8, q9              @ encoding: [0xf2,0x08,0x50,0xf2]
	%tmp3 = and <8 x i16> %tmp1, %tmp2
	%tmp4 = icmp ne <8 x i16> %tmp3, zeroinitializer
  %tmp5 = sext <8 x i1> %tmp4 to <8 x i16>
	ret <8 x i16> %tmp5
}

; CHECK: vtst_4xi32
define <4 x i32> @vtst_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vtst.32	q8, q8, q9              @ encoding: [0xf2,0x08,0x60,0xf2]
	%tmp3 = and <4 x i32> %tmp1, %tmp2
	%tmp4 = icmp ne <4 x i32> %tmp3, zeroinitializer
  %tmp5 = sext <4 x i1> %tmp4 to <4 x i32>
	ret <4 x i32> %tmp5
}
