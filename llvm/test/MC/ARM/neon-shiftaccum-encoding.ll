; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

define <8 x i8> @vsras8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vsra.s8	d16, d17, #8            @ encoding: [0x31,0x01,0xc8,0xf2]
	%tmp3 = ashr <8 x i8> %tmp2, < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
        %tmp4 = add <8 x i8> %tmp1, %tmp3
	ret <8 x i8> %tmp4
}

define <4 x i16> @vsras16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vsra.s16	d16, d17, #16   @ encoding: [0x31,0x01,0xd0,0xf2]
	%tmp3 = ashr <4 x i16> %tmp2, < i16 16, i16 16, i16 16, i16 16 >
        %tmp4 = add <4 x i16> %tmp1, %tmp3
	ret <4 x i16> %tmp4
}

define <2 x i32> @vsras32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vsra.s32	d16, d17, #32   @ encoding: [0x31,0x01,0xe0,0xf2]
	%tmp3 = ashr <2 x i32> %tmp2, < i32 32, i32 32 >
        %tmp4 = add <2 x i32> %tmp1, %tmp3
	ret <2 x i32> %tmp4
}

define <1 x i64> @vsras64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
; CHECK: vsra.s64	d16, d17, #64   @ encoding: [0xb1,0x01,0xc0,0xf2]
	%tmp3 = ashr <1 x i64> %tmp2, < i64 64 >
        %tmp4 = add <1 x i64> %tmp1, %tmp3
	ret <1 x i64> %tmp4
}

define <16 x i8> @vsraQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vsra.s8	q9, q8, #8              @ encoding: [0x70,0x21,0xc8,0xf2]
	%tmp3 = ashr <16 x i8> %tmp2, < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
        %tmp4 = add <16 x i8> %tmp1, %tmp3
	ret <16 x i8> %tmp4
}

define <8 x i16> @vsraQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vsra.s16	q9, q8, #16     @ encoding: [0x70,0x21,0xd0,0xf2]
	%tmp3 = ashr <8 x i16> %tmp2, < i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16 >
        %tmp4 = add <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @vsraQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vsra.s32	q9, q8, #32     @ encoding: [0x70,0x21,0xe0,0xf2]
	%tmp3 = ashr <4 x i32> %tmp2, < i32 32, i32 32, i32 32, i32 32 >
        %tmp4 = add <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

define <2 x i64> @vsraQs64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
; CHECK: vsra.s64	q9, q8, #64     @ encoding: [0xf0,0x21,0xc0,0xf2]
	%tmp3 = ashr <2 x i64> %tmp2, < i64 64, i64 64 >
        %tmp4 = add <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}

define <8 x i8> @vsrau8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vsra.u8	d16, d17, #8            @ encoding: [0x31,0x01,0xc8,0xf3]
	%tmp3 = lshr <8 x i8> %tmp2, < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
        %tmp4 = add <8 x i8> %tmp1, %tmp3
	ret <8 x i8> %tmp4
}

define <4 x i16> @vsrau16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vsra.u16	d16, d17, #16   @ encoding: [0x31,0x01,0xd0,0xf3]
	%tmp3 = lshr <4 x i16> %tmp2, < i16 16, i16 16, i16 16, i16 16 >
        %tmp4 = add <4 x i16> %tmp1, %tmp3
	ret <4 x i16> %tmp4
}

define <2 x i32> @vsrau32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vsra.u32	d16, d17, #32   @ encoding: [0x31,0x01,0xe0,0xf3]
	%tmp3 = lshr <2 x i32> %tmp2, < i32 32, i32 32 >
        %tmp4 = add <2 x i32> %tmp1, %tmp3
	ret <2 x i32> %tmp4
}

define <1 x i64> @vsrau64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
; CHECK: vsra.u64	d16, d17, #64   @ encoding: [0xb1,0x01,0xc0,0xf3]
	%tmp3 = lshr <1 x i64> %tmp2, < i64 64 >
        %tmp4 = add <1 x i64> %tmp1, %tmp3
	ret <1 x i64> %tmp4
}

define <16 x i8> @vsraQu8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vsra.u8	q9, q8, #8              @ encoding: [0x70,0x21,0xc8,0xf3]
	%tmp3 = lshr <16 x i8> %tmp2, < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
        %tmp4 = add <16 x i8> %tmp1, %tmp3
	ret <16 x i8> %tmp4
}

define <8 x i16> @vsraQu16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vsra.u16	q9, q8, #16     @ encoding: [0x70,0x21,0xd0,0xf3]
	%tmp3 = lshr <8 x i16> %tmp2, < i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16 >
        %tmp4 = add <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @vsraQu32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vsra.u32	q9, q8, #32     @ encoding: [0x70,0x21,0xe0,0xf3]
	%tmp3 = lshr <4 x i32> %tmp2, < i32 32, i32 32, i32 32, i32 32 >
        %tmp4 = add <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

define <2 x i64> @vsraQu64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
; CHECK: vsra.u64	q9, q8, #64     @ encoding: [0xf0,0x21,0xc0,0xf3]
	%tmp3 = lshr <2 x i64> %tmp2, < i64 64, i64 64 >
        %tmp4 = add <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}

define <8 x i8> @vrsras8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vrsra.s8	d16, d17, #8    @ encoding: [0x31,0x03,0xc8,0xf2]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vrshifts.v8i8(<8 x i8> %tmp2, <8 x i8> < i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8 >)
        %tmp4 = add <8 x i8> %tmp1, %tmp3
	ret <8 x i8> %tmp4
}

define <4 x i16> @vrsras16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vrsra.s16	d16, d17, #16   @ encoding: [0x31,0x03,0xd0,0xf2]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vrshifts.v4i16(<4 x i16> %tmp2, <4 x i16> < i16 -16, i16 -16, i16 -16, i16 -16 >)
        %tmp4 = add <4 x i16> %tmp1, %tmp3
	ret <4 x i16> %tmp4
}

define <2 x i32> @vrsras32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vrsra.s32	d16, d17, #32   @ encoding: [0x31,0x03,0xe0,0xf2]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vrshifts.v2i32(<2 x i32> %tmp2, <2 x i32> < i32 -32, i32 -32 >)
        %tmp4 = add <2 x i32> %tmp1, %tmp3
	ret <2 x i32> %tmp4
}

define <1 x i64> @vrsras64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
; CHECK: vrsra.s64	d16, d17, #64   @ encoding: [0xb1,0x03,0xc0,0xf2]
	%tmp3 = call <1 x i64> @llvm.arm.neon.vrshifts.v1i64(<1 x i64> %tmp2, <1 x i64> < i64 -64 >)
        %tmp4 = add <1 x i64> %tmp1, %tmp3
	ret <1 x i64> %tmp4
}

define <8 x i8> @vrsrau8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vrsra.u8	d16, d17, #8    @ encoding: [0x31,0x03,0xc8,0xf3]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vrshiftu.v8i8(<8 x i8> %tmp2, <8 x i8> < i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8 >)
        %tmp4 = add <8 x i8> %tmp1, %tmp3
	ret <8 x i8> %tmp4
}

define <4 x i16> @vrsrau16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vrsra.u16	d16, d17, #16   @ encoding: [0x31,0x03,0xd0,0xf3]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vrshiftu.v4i16(<4 x i16> %tmp2, <4 x i16> < i16 -16, i16 -16, i16 -16, i16 -16 >)
        %tmp4 = add <4 x i16> %tmp1, %tmp3
	ret <4 x i16> %tmp4
}

define <2 x i32> @vrsrau32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vrsra.u32	d16, d17, #32   @ encoding: [0x31,0x03,0xe0,0xf3]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vrshiftu.v2i32(<2 x i32> %tmp2, <2 x i32> < i32 -32, i32 -32 >)
        %tmp4 = add <2 x i32> %tmp1, %tmp3
	ret <2 x i32> %tmp4
}

define <1 x i64> @vrsrau64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
; CHECK: vrsra.u64	d16, d17, #64   @ encoding: [0xb1,0x03,0xc0,0xf3]
	%tmp3 = call <1 x i64> @llvm.arm.neon.vrshiftu.v1i64(<1 x i64> %tmp2, <1 x i64> < i64 -64 >)
        %tmp4 = add <1 x i64> %tmp1, %tmp3
	ret <1 x i64> %tmp4
}

define <16 x i8> @vrsraQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vrsra.s8	q9, q8, #8      @ encoding: [0x70,0x23,0xc8,0xf2]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vrshifts.v16i8(<16 x i8> %tmp2, <16 x i8> < i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8 >)
        %tmp4 = add <16 x i8> %tmp1, %tmp3
	ret <16 x i8> %tmp4
}

define <8 x i16> @vrsraQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vrsra.s16	q9, q8, #16     @ encoding: [0x70,0x23,0xd0,0xf2]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vrshifts.v8i16(<8 x i16> %tmp2, <8 x i16> < i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16 >)
        %tmp4 = add <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @vrsraQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vrsra.s32	q9, q8, #32     @ encoding: [0x70,0x23,0xe0,0xf2]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vrshifts.v4i32(<4 x i32> %tmp2, <4 x i32> < i32 -32, i32 -32, i32 -32, i32 -32 >)
        %tmp4 = add <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

define <2 x i64> @vrsraQs64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
; CHECK: vrsra.s64	q9, q8, #64     @ encoding: [0xf0,0x23,0xc0,0xf2]
	%tmp3 = call <2 x i64> @llvm.arm.neon.vrshifts.v2i64(<2 x i64> %tmp2, <2 x i64> < i64 -64, i64 -64 >)
        %tmp4 = add <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}

define <16 x i8> @vrsraQu8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vrsra.u8	q9, q8, #8      @ encoding: [0x70,0x23,0xc8,0xf3]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vrshiftu.v16i8(<16 x i8> %tmp2, <16 x i8> < i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8 >)
        %tmp4 = add <16 x i8> %tmp1, %tmp3
	ret <16 x i8> %tmp4
}

define <8 x i16> @vrsraQu16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vrsra.u16	q9, q8, #16     @ encoding: [0x70,0x23,0xd0,0xf3]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vrshiftu.v8i16(<8 x i16> %tmp2, <8 x i16> < i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16 >)
        %tmp4 = add <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @vrsraQu32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vrsra.u32	q9, q8, #32     @ encoding: [0x70,0x23,0xe0,0xf3]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vrshiftu.v4i32(<4 x i32> %tmp2, <4 x i32> < i32 -32, i32 -32, i32 -32, i32 -32 >)
        %tmp4 = add <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

define <2 x i64> @vrsraQu64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
; CHECK: vrsra.u64	q9, q8, #64     @ encoding: [0xf0,0x23,0xc0,0xf3]
	%tmp3 = call <2 x i64> @llvm.arm.neon.vrshiftu.v2i64(<2 x i64> %tmp2, <2 x i64> < i64 -64, i64 -64 >)
        %tmp4 = add <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}

declare <8 x i8>  @llvm.arm.neon.vrshifts.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vrshifts.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vrshifts.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vrshifts.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vrshiftu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vrshiftu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vrshiftu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vrshiftu.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vrshifts.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vrshifts.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vrshifts.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vrshifts.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vrshiftu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vrshiftu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vrshiftu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vrshiftu.v2i64(<2 x i64>, <2 x i64>) nounwind readnone
