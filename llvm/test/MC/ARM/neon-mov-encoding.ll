; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

; CHECK: vmov_8xi8
define <8 x i8> @vmov_8xi8() nounwind {
; CHECK: vmov.i8	d16, #0x8               @ encoding: [0x18,0x0e,0xc0,0xf2]
	ret <8 x i8> < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
}

; CHECK: vmov_4xi16a
define <4 x i16> @vmov_4xi16a() nounwind {
; CHECK: vmov.i16	d16, #0x10      @ encoding: [0x10,0x08,0xc1,0xf2]
	ret <4 x i16> < i16 16, i16 16, i16 16, i16 16 >
}

; CHECK: vmov_4xi16b
define <4 x i16> @vmov_4xi16b() nounwind {
; CHECK: vmov.i16	d16, #0x1000    @ encoding: [0x10,0x0a,0xc1,0xf2]
	ret <4 x i16> < i16 4096, i16 4096, i16 4096, i16 4096 >
}

; CHECK: vmov_2xi32a
define <2 x i32> @vmov_2xi32a() nounwind {
; CHECK: vmov.i32	d16, #0x20      @ encoding: [0x10,0x00,0xc2,0xf2]
	ret <2 x i32> < i32 32, i32 32 >
}

; CHECK: vmov_2xi32b
define <2 x i32> @vmov_2xi32b() nounwind {
; CHECK: vmov.i32	d16, #0x2000    @ encoding: [0x10,0x02,0xc2,0xf2]
	ret <2 x i32> < i32 8192, i32 8192 >
}

; CHECK: vmov_2xi32c
define <2 x i32> @vmov_2xi32c() nounwind {
; CHECK: vmov.i32	d16, #0x200000  @ encoding: [0x10,0x04,0xc2,0xf2]
	ret <2 x i32> < i32 2097152, i32 2097152 >
}

; CHECK: vmov_2xi32d
define <2 x i32> @vmov_2xi32d() nounwind {
; CHECK: vmov.i32	d16, #0x20000000 @ encoding: [0x10,0x06,0xc2,0xf2]
	ret <2 x i32> < i32 536870912, i32 536870912 >
}

; CHECK: vmov_2xi32e
define <2 x i32> @vmov_2xi32e() nounwind {
; CHECK: vmov.i32	d16, #0x20FF    @ encoding: [0x10,0x0c,0xc2,0xf2]
	ret <2 x i32> < i32 8447, i32 8447 >
}

; CHECK: vmov_2xi32f
define <2 x i32> @vmov_2xi32f() nounwind {
; CHECK: vmov.i32	d16, #0x20FFFF  @ encoding: [0x10,0x0d,0xc2,0xf2]
	ret <2 x i32> < i32 2162687, i32 2162687 >
}

; CHECK: vmov_1xi64
define <1 x i64> @vmov_1xi64() nounwind {
; CHECK: vmov.i64	d16, #0xFF0000FF0000FFFF @ encoding: [0x33,0x0e,0xc1,0xf3]
	ret <1 x i64> < i64 18374687574888349695 >
}

; CHECK: vmov_16xi8
define <16 x i8> @vmov_16xi8() nounwind {
; CHECK: vmov.i8	q8, #0x8                @ encoding: [0x58,0x0e,0xc0,0xf2]
	ret <16 x i8> < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
}

; CHECK: vmov_8xi16a
define <8 x i16> @vmov_8xi16a() nounwind {
; CHECK: vmov.i16	q8, #0x10       @ encoding: [0x50,0x08,0xc1,0xf2]
	ret <8 x i16> < i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16 >
}

; CHECK: vmov_8xi16b
define <8 x i16> @vmov_8xi16b() nounwind {
; CHECK: vmov.i16	q8, #0x1000     @ encoding: [0x50,0x0a,0xc1,0xf2]
	ret <8 x i16> < i16 4096, i16 4096, i16 4096, i16 4096, i16 4096, i16 4096, i16 4096, i16 4096 >
}

; CHECK: vmov_4xi32a
define <4 x i32> @vmov_4xi32a() nounwind {
; CHECK: vmov.i32	q8, #0x20       @ encoding: [0x50,0x00,0xc2,0xf2]
	ret <4 x i32> < i32 32, i32 32, i32 32, i32 32 >
}

; CHECK: vmov_4xi32b
define <4 x i32> @vmov_4xi32b() nounwind {
; CHECK: vmov.i32	q8, #0x2000     @ encoding: [0x50,0x02,0xc2,0xf2]
	ret <4 x i32> < i32 8192, i32 8192, i32 8192, i32 8192 >
}

; CHECK: vmov_4xi32c
define <4 x i32> @vmov_4xi32c() nounwind {
; CHECK: vmov.i32	q8, #0x200000   @ encoding: [0x50,0x04,0xc2,0xf2]
	ret <4 x i32> < i32 2097152, i32 2097152, i32 2097152, i32 2097152 >
}

; CHECK: vmov_4xi32d
define <4 x i32> @vmov_4xi32d() nounwind {
; CHECK: vmov.i32	q8, #0x20000000 @ encoding: [0x50,0x06,0xc2,0xf2]
	ret <4 x i32> < i32 536870912, i32 536870912, i32 536870912, i32 536870912 >
}

; CHECK: vmov_4xi32e
define <4 x i32> @vmov_4xi32e() nounwind {
; CHECK: vmov.i32	q8, #0x20FF     @ encoding: [0x50,0x0c,0xc2,0xf2]
	ret <4 x i32> < i32 8447, i32 8447, i32 8447, i32 8447 >
}

; CHECK: vmov_4xi32f
define <4 x i32> @vmov_4xi32f() nounwind {
; CHECK: vmov.i32	q8, #0x20FFFF   @ encoding: [0x50,0x0d,0xc2,0xf2]
	ret <4 x i32> < i32 2162687, i32 2162687, i32 2162687, i32 2162687 >
}

; CHECK: vmov_2xi64
define <2 x i64> @vmov_2xi64() nounwind {
; CHECK: vmov.i64	q8, #0xFF0000FF0000FFFF @ encoding: [0x73,0x0e,0xc1,0xf3]
	ret <2 x i64> < i64 18374687574888349695, i64 18374687574888349695 >
}

; CHECK: vmvn_4xi16a
define <4 x i16> @vmvn_4xi16a() nounwind {
; CHECK: vmvn.i16	d16, #0x10      @ encoding: [0x30,0x08,0xc1,0xf2]
	ret <4 x i16> < i16 65519, i16 65519, i16 65519, i16 65519 >
}

; CHECK: vmvn_4xi16b
define <4 x i16> @vmvn_4xi16b() nounwind {
; CHECK: vmvn.i16	d16, #0x1000    @ encoding: [0x30,0x0a,0xc1,0xf2]
	ret <4 x i16> < i16 61439, i16 61439, i16 61439, i16 61439 >
}

; CHECK: vmvn_2xi32a
define <2 x i32> @vmvn_2xi32a() nounwind {
; CHECK: vmvn.i32	d16, #0x20      @ encoding: [0x30,0x00,0xc2,0xf2]
	ret <2 x i32> < i32 4294967263, i32 4294967263 >
}

; CHECK: vmvn_2xi32b
define <2 x i32> @vmvn_2xi32b() nounwind {
; CHECK: vmvn.i32	d16, #0x2000    @ encoding: [0x30,0x02,0xc2,0xf2]
	ret <2 x i32> < i32 4294959103, i32 4294959103 >
}

; CHECK: vmvn_2xi32c
define <2 x i32> @vmvn_2xi32c() nounwind {
; CHECK: vmvn.i32	d16, #0x200000  @ encoding: [0x30,0x04,0xc2,0xf2]
	ret <2 x i32> < i32 4292870143, i32 4292870143 >
}

; CHECK: vmvn_2xi32d
define <2 x i32> @vmvn_2xi32d() nounwind {
; CHECK: vmvn.i32	d16, #0x20000000 @ encoding: [0x30,0x06,0xc2,0xf2]
	ret <2 x i32> < i32 3758096383, i32 3758096383 >
}

; CHECK: vmvn_2xi32e
define <2 x i32> @vmvn_2xi32e() nounwind {
; CHECK: vmvn.i32	d16, #0x20FF    @ encoding: [0x30,0x0c,0xc2,0xf2]
	ret <2 x i32> < i32 4294958848, i32 4294958848 >
}

; CHECK: vmvn_2xi32f
define <2 x i32> @vmvn_2xi32f() nounwind {
; CHECK: vmvn.i32	d16, #0x20FFFF  @ encoding: [0x30,0x0d,0xc2,0xf2]
	ret <2 x i32> < i32 4292804608, i32 4292804608 >
}

define <8 x i16> @vmovls8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vmovl.s8	q8, d16         @ encoding: [0x30,0x0a,0xc8,0xf2]
	%tmp2 = sext <8 x i8> %tmp1 to <8 x i16>
	ret <8 x i16> %tmp2
}

define <4 x i32> @vmovls16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vmovl.s16	q8, d16         @ encoding: [0x30,0x0a,0xd0,0xf2]
	%tmp2 = sext <4 x i16> %tmp1 to <4 x i32>
	ret <4 x i32> %tmp2
}

define <2 x i64> @vmovls32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vmovl.s32	q8, d16         @ encoding: [0x30,0x0a,0xe0,0xf2]
	%tmp2 = sext <2 x i32> %tmp1 to <2 x i64>
	ret <2 x i64> %tmp2
}

define <8 x i16> @vmovlu8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vmovl.u8	q8, d16         @ encoding: [0x30,0x0a,0xc8,0xf3]
	%tmp2 = zext <8 x i8> %tmp1 to <8 x i16>
	ret <8 x i16> %tmp2
}

define <4 x i32> @vmovlu16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vmovl.u16	q8, d16         @ encoding: [0x30,0x0a,0xd0,0xf3]
	%tmp2 = zext <4 x i16> %tmp1 to <4 x i32>
	ret <4 x i32> %tmp2
}

define <2 x i64> @vmovlu32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vmovl.u32	q8, d16         @ encoding: [0x30,0x0a,0xe0,0xf3]
	%tmp2 = zext <2 x i32> %tmp1 to <2 x i64>
	ret <2 x i64> %tmp2
}

define <8 x i8> @vmovni16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vmovn.i16	d16, q8         @ encoding: [0x20,0x02,0xf2,0xf3]
	%tmp2 = trunc <8 x i16> %tmp1 to <8 x i8>
	ret <8 x i8> %tmp2
}

define <4 x i16> @vmovni32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vmovn.i32	d16, q8         @ encoding: [0x20,0x02,0xf6,0xf3]
	%tmp2 = trunc <4 x i32> %tmp1 to <4 x i16>
	ret <4 x i16> %tmp2
}

define <2 x i32> @vmovni64(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
; CHECK: vmovn.i64	d16, q8         @ encoding: [0x20,0x02,0xfa,0xf3]
	%tmp2 = trunc <2 x i64> %tmp1 to <2 x i32>
	ret <2 x i32> %tmp2
}

define <8 x i8> @vqmovns16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vqmovn.s16	d16, q8         @ encoding: [0xa0,0x02,0xf2,0xf3]
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqmovns.v8i8(<8 x i16> %tmp1)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqmovns32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vqmovn.s32	d16, q8         @ encoding: [0xa0,0x02,0xf6,0xf3]
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqmovns.v4i16(<4 x i32> %tmp1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqmovns64(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
; CHECK: vqmovn.s64	d16, q8         @ encoding: [0xa0,0x02,0xfa,0xf3]
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqmovns.v2i32(<2 x i64> %tmp1)
	ret <2 x i32> %tmp2
}

define <8 x i8> @vqmovnu16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vqmovn.u16	d16, q8         @ encoding: [0xe0,0x02,0xf2,0xf3]
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqmovnu.v8i8(<8 x i16> %tmp1)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqmovnu32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vqmovn.u32	d16, q8         @ encoding: [0xe0,0x02,0xf6,0xf3]
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqmovnu.v4i16(<4 x i32> %tmp1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqmovnu64(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
; CHECK: vqmovn.u64	d16, q8         @ encoding: [0xe0,0x02,0xfa,0xf3]
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqmovnu.v2i32(<2 x i64> %tmp1)
	ret <2 x i32> %tmp2
}

define <8 x i8> @vqmovuns16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vqmovun.s16	d16, q8         @ encoding: [0x60,0x02,0xf2,0xf3]
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqmovnsu.v8i8(<8 x i16> %tmp1)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqmovuns32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vqmovun.s32	d16, q8         @ encoding: [0x60,0x02,0xf6,0xf3]
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqmovnsu.v4i16(<4 x i32> %tmp1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqmovuns64(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
; CHECK: vqmovun.s64	d16, q8         @ encoding: [0x60,0x02,0xfa,0xf3]
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqmovnsu.v2i32(<2 x i64> %tmp1)
	ret <2 x i32> %tmp2
}

declare <8 x i8>  @llvm.arm.neon.vqmovns.v8i8(<8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqmovns.v4i16(<4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqmovns.v2i32(<2 x i64>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vqmovnu.v8i8(<8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqmovnu.v4i16(<4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqmovnu.v2i32(<2 x i64>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vqmovnsu.v8i8(<8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqmovnsu.v4i16(<4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqmovnsu.v2i32(<2 x i64>) nounwind readnone

define i32 @vget_lanes8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vmov.s8	r0, d16[1]              @ encoding: [0xb0,0x0b,0x50,0xee]
	%tmp2 = extractelement <8 x i8> %tmp1, i32 1
	%tmp3 = sext i8 %tmp2 to i32
	ret i32 %tmp3
}

define i32 @vget_lanes16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vmov.s16	r0, d16[1]      @ encoding: [0xf0,0x0b,0x10,0xee]
	%tmp2 = extractelement <4 x i16> %tmp1, i32 1
	%tmp3 = sext i16 %tmp2 to i32
	ret i32 %tmp3
}

define i32 @vget_laneu8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vmov.u8	r0, d16[1]              @ encoding: [0xb0,0x0b,0xd0,0xee]
	%tmp2 = extractelement <8 x i8> %tmp1, i32 1
	%tmp3 = zext i8 %tmp2 to i32
	ret i32 %tmp3
}

define i32 @vget_laneu16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vmov.u16	r0, d16[1]      @ encoding: [0xf0,0x0b,0x90,0xee]
	%tmp2 = extractelement <4 x i16> %tmp1, i32 1
	%tmp3 = zext i16 %tmp2 to i32
	ret i32 %tmp3
}

; Do a vector add to keep the extraction from being done directly from memory.
define i32 @vget_lanei32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = add <2 x i32> %tmp1, %tmp1
; CHECK: vmov.32	r0, d16[1]              @ encoding: [0x90,0x0b,0x30,0xee]
	%tmp3 = extractelement <2 x i32> %tmp2, i32 1
	ret i32 %tmp3
}

define i32 @vgetQ_lanes8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
; CHECK: vmov.s8	r0, d16[1]              @ encoding: [0xb0,0x0b,0x50,0xee]
	%tmp2 = extractelement <16 x i8> %tmp1, i32 1
	%tmp3 = sext i8 %tmp2 to i32
	ret i32 %tmp3
}

define i32 @vgetQ_lanes16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vmov.s16	r0, d16[1]      @ encoding: [0xf0,0x0b,0x10,0xee]
	%tmp2 = extractelement <8 x i16> %tmp1, i32 1
	%tmp3 = sext i16 %tmp2 to i32
	ret i32 %tmp3
}

define i32 @vgetQ_laneu8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
; CHECK: vmov.u8	r0, d16[1]              @ encoding: [0xb0,0x0b,0xd0,0xee]
	%tmp2 = extractelement <16 x i8> %tmp1, i32 1
	%tmp3 = zext i8 %tmp2 to i32
	ret i32 %tmp3
}

define i32 @vgetQ_laneu16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vmov.u16	r0, d16[1]      @ encoding: [0xf0,0x0b,0x90,0xee]
	%tmp2 = extractelement <8 x i16> %tmp1, i32 1
	%tmp3 = zext i16 %tmp2 to i32
	ret i32 %tmp3
}

; Do a vector add to keep the extraction from being done directly from memory.
define i32 @vgetQ_lanei32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = add <4 x i32> %tmp1, %tmp1
; CHECK: vmov.32	r0, d16[1]              @ encoding: [0x90,0x0b,0x30,0xee]
	%tmp3 = extractelement <4 x i32> %tmp2, i32 1
	ret i32 %tmp3
}

define <8 x i8> @vset_lane8(<8 x i8>* %A, i8 %B) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vmov.8	d16[1], r1              @ encoding: [0xb0,0x1b,0x40,0xee]
	%tmp2 = insertelement <8 x i8> %tmp1, i8 %B, i32 1
	ret <8 x i8> %tmp2
}

define <4 x i16> @vset_lane16(<4 x i16>* %A, i16 %B) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vmov.16	d16[1], r1              @ encoding: [0xf0,0x1b,0x00,0xee
	%tmp2 = insertelement <4 x i16> %tmp1, i16 %B, i32 1
	ret <4 x i16> %tmp2
}

define <2 x i32> @vset_lane32(<2 x i32>* %A, i32 %B) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vmov.32	d16[1], r1              @ encoding: [0x90,0x1b,0x20,0xee]
	%tmp2 = insertelement <2 x i32> %tmp1, i32 %B, i32 1
	ret <2 x i32> %tmp2
}

define <16 x i8> @vsetQ_lane8(<16 x i8>* %A, i8 %B) nounwind {
	%tmp1 = load <16 x i8>* %A
; CHECK: vmov.8	d18[1], r1              @ encoding: [0xb0,0x1b,0x42,0xee]
	%tmp2 = insertelement <16 x i8> %tmp1, i8 %B, i32 1
	ret <16 x i8> %tmp2
}

define <8 x i16> @vsetQ_lane16(<8 x i16>* %A, i16 %B) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vmov.16	d18[1], r1              @ encoding: [0xf0,0x1b,0x02,0xee]
	%tmp2 = insertelement <8 x i16> %tmp1, i16 %B, i32 1
	ret <8 x i16> %tmp2
}

define <4 x i32> @vsetQ_lane32(<4 x i32>* %A, i32 %B) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vmov.32	d18[1], r1              @ encoding: [0x90,0x1b,0x22,0xee]
	%tmp2 = insertelement <4 x i32> %tmp1, i32 %B, i32 1
	ret <4 x i32> %tmp2
}
