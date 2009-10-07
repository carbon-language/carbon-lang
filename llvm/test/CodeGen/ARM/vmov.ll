; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @v_movi8() nounwind {
;CHECK: v_movi8:
;CHECK: vmov.i8
	ret <8 x i8> < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
}

define <4 x i16> @v_movi16a() nounwind {
;CHECK: v_movi16a:
;CHECK: vmov.i16
	ret <4 x i16> < i16 16, i16 16, i16 16, i16 16 >
}

; 0x1000 = 4096
define <4 x i16> @v_movi16b() nounwind {
;CHECK: v_movi16b:
;CHECK: vmov.i16
	ret <4 x i16> < i16 4096, i16 4096, i16 4096, i16 4096 >
}

define <2 x i32> @v_movi32a() nounwind {
;CHECK: v_movi32a:
;CHECK: vmov.i32
	ret <2 x i32> < i32 32, i32 32 >
}

; 0x2000 = 8192
define <2 x i32> @v_movi32b() nounwind {
;CHECK: v_movi32b:
;CHECK: vmov.i32
	ret <2 x i32> < i32 8192, i32 8192 >
}

; 0x200000 = 2097152
define <2 x i32> @v_movi32c() nounwind {
;CHECK: v_movi32c:
;CHECK: vmov.i32
	ret <2 x i32> < i32 2097152, i32 2097152 >
}

; 0x20000000 = 536870912
define <2 x i32> @v_movi32d() nounwind {
;CHECK: v_movi32d:
;CHECK: vmov.i32
	ret <2 x i32> < i32 536870912, i32 536870912 >
}

; 0x20ff = 8447
define <2 x i32> @v_movi32e() nounwind {
;CHECK: v_movi32e:
;CHECK: vmov.i32
	ret <2 x i32> < i32 8447, i32 8447 >
}

; 0x20ffff = 2162687
define <2 x i32> @v_movi32f() nounwind {
;CHECK: v_movi32f:
;CHECK: vmov.i32
	ret <2 x i32> < i32 2162687, i32 2162687 >
}

; 0xff0000ff0000ffff = 18374687574888349695
define <1 x i64> @v_movi64() nounwind {
;CHECK: v_movi64:
;CHECK: vmov.i64
	ret <1 x i64> < i64 18374687574888349695 >
}

define <16 x i8> @v_movQi8() nounwind {
;CHECK: v_movQi8:
;CHECK: vmov.i8
	ret <16 x i8> < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
}

define <8 x i16> @v_movQi16a() nounwind {
;CHECK: v_movQi16a:
;CHECK: vmov.i16
	ret <8 x i16> < i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16 >
}

; 0x1000 = 4096
define <8 x i16> @v_movQi16b() nounwind {
;CHECK: v_movQi16b:
;CHECK: vmov.i16
	ret <8 x i16> < i16 4096, i16 4096, i16 4096, i16 4096, i16 4096, i16 4096, i16 4096, i16 4096 >
}

define <4 x i32> @v_movQi32a() nounwind {
;CHECK: v_movQi32a:
;CHECK: vmov.i32
	ret <4 x i32> < i32 32, i32 32, i32 32, i32 32 >
}

; 0x2000 = 8192
define <4 x i32> @v_movQi32b() nounwind {
;CHECK: v_movQi32b:
;CHECK: vmov.i32
	ret <4 x i32> < i32 8192, i32 8192, i32 8192, i32 8192 >
}

; 0x200000 = 2097152
define <4 x i32> @v_movQi32c() nounwind {
;CHECK: v_movQi32c:
;CHECK: vmov.i32
	ret <4 x i32> < i32 2097152, i32 2097152, i32 2097152, i32 2097152 >
}

; 0x20000000 = 536870912
define <4 x i32> @v_movQi32d() nounwind {
;CHECK: v_movQi32d:
;CHECK: vmov.i32
	ret <4 x i32> < i32 536870912, i32 536870912, i32 536870912, i32 536870912 >
}

; 0x20ff = 8447
define <4 x i32> @v_movQi32e() nounwind {
;CHECK: v_movQi32e:
;CHECK: vmov.i32
	ret <4 x i32> < i32 8447, i32 8447, i32 8447, i32 8447 >
}

; 0x20ffff = 2162687
define <4 x i32> @v_movQi32f() nounwind {
;CHECK: v_movQi32f:
;CHECK: vmov.i32
	ret <4 x i32> < i32 2162687, i32 2162687, i32 2162687, i32 2162687 >
}

; 0xff0000ff0000ffff = 18374687574888349695
define <2 x i64> @v_movQi64() nounwind {
;CHECK: v_movQi64:
;CHECK: vmov.i64
	ret <2 x i64> < i64 18374687574888349695, i64 18374687574888349695 >
}
