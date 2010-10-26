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
