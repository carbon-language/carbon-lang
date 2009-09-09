; RUN: llc < %s -march=arm -mattr=+neon > %t
; RUN: grep vmov.i8 %t | count 2
; RUN: grep vmov.i16 %t | count 4
; RUN: grep vmov.i32 %t | count 12
; RUN: grep vmov.i64 %t | count 2
; Note: function names do not include "vmov" to allow simple grep for opcodes

define <8 x i8> @v_movi8() nounwind {
	ret <8 x i8> < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
}

define <4 x i16> @v_movi16a() nounwind {
	ret <4 x i16> < i16 16, i16 16, i16 16, i16 16 >
}

; 0x1000 = 4096
define <4 x i16> @v_movi16b() nounwind {
	ret <4 x i16> < i16 4096, i16 4096, i16 4096, i16 4096 >
}

define <2 x i32> @v_movi32a() nounwind {
	ret <2 x i32> < i32 32, i32 32 >
}

; 0x2000 = 8192
define <2 x i32> @v_movi32b() nounwind {
	ret <2 x i32> < i32 8192, i32 8192 >
}

; 0x200000 = 2097152
define <2 x i32> @v_movi32c() nounwind {
	ret <2 x i32> < i32 2097152, i32 2097152 >
}

; 0x20000000 = 536870912
define <2 x i32> @v_movi32d() nounwind {
	ret <2 x i32> < i32 536870912, i32 536870912 >
}

; 0x20ff = 8447
define <2 x i32> @v_movi32e() nounwind {
	ret <2 x i32> < i32 8447, i32 8447 >
}

; 0x20ffff = 2162687
define <2 x i32> @v_movi32f() nounwind {
	ret <2 x i32> < i32 2162687, i32 2162687 >
}

; 0xff0000ff0000ffff = 18374687574888349695
define <1 x i64> @v_movi64() nounwind {
	ret <1 x i64> < i64 18374687574888349695 >
}

define <16 x i8> @v_movQi8() nounwind {
	ret <16 x i8> < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
}

define <8 x i16> @v_movQi16a() nounwind {
	ret <8 x i16> < i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16 >
}

; 0x1000 = 4096
define <8 x i16> @v_movQi16b() nounwind {
	ret <8 x i16> < i16 4096, i16 4096, i16 4096, i16 4096, i16 4096, i16 4096, i16 4096, i16 4096 >
}

define <4 x i32> @v_movQi32a() nounwind {
	ret <4 x i32> < i32 32, i32 32, i32 32, i32 32 >
}

; 0x2000 = 8192
define <4 x i32> @v_movQi32b() nounwind {
	ret <4 x i32> < i32 8192, i32 8192, i32 8192, i32 8192 >
}

; 0x200000 = 2097152
define <4 x i32> @v_movQi32c() nounwind {
	ret <4 x i32> < i32 2097152, i32 2097152, i32 2097152, i32 2097152 >
}

; 0x20000000 = 536870912
define <4 x i32> @v_movQi32d() nounwind {
	ret <4 x i32> < i32 536870912, i32 536870912, i32 536870912, i32 536870912 >
}

; 0x20ff = 8447
define <4 x i32> @v_movQi32e() nounwind {
	ret <4 x i32> < i32 8447, i32 8447, i32 8447, i32 8447 >
}

; 0x20ffff = 2162687
define <4 x i32> @v_movQi32f() nounwind {
	ret <4 x i32> < i32 2162687, i32 2162687, i32 2162687, i32 2162687 >
}

; 0xff0000ff0000ffff = 18374687574888349695
define <2 x i64> @v_movQi64() nounwind {
	ret <2 x i64> < i64 18374687574888349695, i64 18374687574888349695 >
}

