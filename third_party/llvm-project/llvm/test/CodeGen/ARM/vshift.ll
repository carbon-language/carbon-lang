; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

define <8 x i8> @vshls8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: vshls8:
;CHECK: vshl.u8
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = shl <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vshls16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: vshls16:
;CHECK: vshl.u16
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = shl <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vshls32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vshls32:
;CHECK: vshl.u32
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = shl <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <1 x i64> @vshls64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK-LABEL: vshls64:
;CHECK: vshl.u64
	%tmp1 = load <1 x i64>, <1 x i64>* %A
	%tmp2 = load <1 x i64>, <1 x i64>* %B
	%tmp3 = shl <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

define <8 x i8> @vshli8(<8 x i8>* %A) nounwind {
;CHECK-LABEL: vshli8:
;CHECK: vshl.i8
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = shl <8 x i8> %tmp1, < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >
	ret <8 x i8> %tmp2
}

define <4 x i16> @vshli16(<4 x i16>* %A) nounwind {
;CHECK-LABEL: vshli16:
;CHECK: vshl.i16
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = shl <4 x i16> %tmp1, < i16 15, i16 15, i16 15, i16 15 >
	ret <4 x i16> %tmp2
}

define <2 x i32> @vshli32(<2 x i32>* %A) nounwind {
;CHECK-LABEL: vshli32:
;CHECK: vshl.i32
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = shl <2 x i32> %tmp1, < i32 31, i32 31 >
	ret <2 x i32> %tmp2
}

define <1 x i64> @vshli64(<1 x i64>* %A) nounwind {
;CHECK-LABEL: vshli64:
;CHECK: vshl.i64
	%tmp1 = load <1 x i64>, <1 x i64>* %A
	%tmp2 = shl <1 x i64> %tmp1, < i64 63 >
	ret <1 x i64> %tmp2
}

define <16 x i8> @vshlQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: vshlQs8:
;CHECK: vshl.u8
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = shl <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vshlQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: vshlQs16:
;CHECK: vshl.u16
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = shl <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vshlQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: vshlQs32:
;CHECK: vshl.u32
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = shl <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <2 x i64> @vshlQs64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: vshlQs64:
;CHECK: vshl.u64
	%tmp1 = load <2 x i64>, <2 x i64>* %A
	%tmp2 = load <2 x i64>, <2 x i64>* %B
	%tmp3 = shl <2 x i64> %tmp1, %tmp2
	ret <2 x i64> %tmp3
}

define <16 x i8> @vshlQi8(<16 x i8>* %A) nounwind {
;CHECK-LABEL: vshlQi8:
;CHECK: vshl.i8
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = shl <16 x i8> %tmp1, < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >
	ret <16 x i8> %tmp2
}

define <8 x i16> @vshlQi16(<8 x i16>* %A) nounwind {
;CHECK-LABEL: vshlQi16:
;CHECK: vshl.i16
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = shl <8 x i16> %tmp1, < i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15 >
	ret <8 x i16> %tmp2
}

define <4 x i32> @vshlQi32(<4 x i32>* %A) nounwind {
;CHECK-LABEL: vshlQi32:
;CHECK: vshl.i32
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = shl <4 x i32> %tmp1, < i32 31, i32 31, i32 31, i32 31 >
	ret <4 x i32> %tmp2
}

define <2 x i64> @vshlQi64(<2 x i64>* %A) nounwind {
;CHECK-LABEL: vshlQi64:
;CHECK: vshl.i64
	%tmp1 = load <2 x i64>, <2 x i64>* %A
	%tmp2 = shl <2 x i64> %tmp1, < i64 63, i64 63 >
	ret <2 x i64> %tmp2
}

define <8 x i8> @vlshru8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: vlshru8:
;CHECK: vneg.s8
;CHECK: vshl.u8
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = lshr <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vlshru16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: vlshru16:
;CHECK: vneg.s16
;CHECK: vshl.u16
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = lshr <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vlshru32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vlshru32:
;CHECK: vneg.s32
;CHECK: vshl.u32
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = lshr <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <1 x i64> @vlshru64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK-LABEL: vlshru64:
;CHECK: vsub.i64
;CHECK: vshl.u64
	%tmp1 = load <1 x i64>, <1 x i64>* %A
	%tmp2 = load <1 x i64>, <1 x i64>* %B
	%tmp3 = lshr <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

define <8 x i8> @vlshri8(<8 x i8>* %A) nounwind {
;CHECK-LABEL: vlshri8:
;CHECK: vshr.u8
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = lshr <8 x i8> %tmp1, < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >
	ret <8 x i8> %tmp2
}

define <4 x i16> @vlshri16(<4 x i16>* %A) nounwind {
;CHECK-LABEL: vlshri16:
;CHECK: vshr.u16
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = lshr <4 x i16> %tmp1, < i16 15, i16 15, i16 15, i16 15 >
	ret <4 x i16> %tmp2
}

define <2 x i32> @vlshri32(<2 x i32>* %A) nounwind {
;CHECK-LABEL: vlshri32:
;CHECK: vshr.u32
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = lshr <2 x i32> %tmp1, < i32 31, i32 31 >
	ret <2 x i32> %tmp2
}

define <1 x i64> @vlshri64(<1 x i64>* %A) nounwind {
;CHECK-LABEL: vlshri64:
;CHECK: vshr.u64
	%tmp1 = load <1 x i64>, <1 x i64>* %A
	%tmp2 = lshr <1 x i64> %tmp1, < i64 63 >
	ret <1 x i64> %tmp2
}

define <16 x i8> @vlshrQu8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: vlshrQu8:
;CHECK: vneg.s8
;CHECK: vshl.u8
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = lshr <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vlshrQu16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: vlshrQu16:
;CHECK: vneg.s16
;CHECK: vshl.u16
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = lshr <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vlshrQu32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: vlshrQu32:
;CHECK: vneg.s32
;CHECK: vshl.u32
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = lshr <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <2 x i64> @vlshrQu64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: vlshrQu64:
;CHECK: vsub.i64
;CHECK: vshl.u64
	%tmp1 = load <2 x i64>, <2 x i64>* %A
	%tmp2 = load <2 x i64>, <2 x i64>* %B
	%tmp3 = lshr <2 x i64> %tmp1, %tmp2
	ret <2 x i64> %tmp3
}

define <16 x i8> @vlshrQi8(<16 x i8>* %A) nounwind {
;CHECK-LABEL: vlshrQi8:
;CHECK: vshr.u8
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = lshr <16 x i8> %tmp1, < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >
	ret <16 x i8> %tmp2
}

define <8 x i16> @vlshrQi16(<8 x i16>* %A) nounwind {
;CHECK-LABEL: vlshrQi16:
;CHECK: vshr.u16
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = lshr <8 x i16> %tmp1, < i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15 >
	ret <8 x i16> %tmp2
}

define <4 x i32> @vlshrQi32(<4 x i32>* %A) nounwind {
;CHECK-LABEL: vlshrQi32:
;CHECK: vshr.u32
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = lshr <4 x i32> %tmp1, < i32 31, i32 31, i32 31, i32 31 >
	ret <4 x i32> %tmp2
}

define <2 x i64> @vlshrQi64(<2 x i64>* %A) nounwind {
;CHECK-LABEL: vlshrQi64:
;CHECK: vshr.u64
	%tmp1 = load <2 x i64>, <2 x i64>* %A
	%tmp2 = lshr <2 x i64> %tmp1, < i64 63, i64 63 >
	ret <2 x i64> %tmp2
}

; Example that requires splitting and expanding a vector shift.
define <2 x i64> @update(<2 x i64> %val) nounwind readnone {
entry:
	%shr = lshr <2 x i64> %val, < i64 2, i64 2 >		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %shr
}

define <8 x i8> @vashrs8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: vashrs8:
;CHECK: vneg.s8
;CHECK: vshl.s8
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = ashr <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vashrs16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: vashrs16:
;CHECK: vneg.s16
;CHECK: vshl.s16
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = ashr <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vashrs32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vashrs32:
;CHECK: vneg.s32
;CHECK: vshl.s32
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = ashr <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <1 x i64> @vashrs64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK-LABEL: vashrs64:
;CHECK: vsub.i64
;CHECK: vshl.s64
	%tmp1 = load <1 x i64>, <1 x i64>* %A
	%tmp2 = load <1 x i64>, <1 x i64>* %B
	%tmp3 = ashr <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

define <8 x i8> @vashri8(<8 x i8>* %A) nounwind {
;CHECK-LABEL: vashri8:
;CHECK: vshr.s8
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = ashr <8 x i8> %tmp1, < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >
	ret <8 x i8> %tmp2
}

define <4 x i16> @vashri16(<4 x i16>* %A) nounwind {
;CHECK-LABEL: vashri16:
;CHECK: vshr.s16
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = ashr <4 x i16> %tmp1, < i16 15, i16 15, i16 15, i16 15 >
	ret <4 x i16> %tmp2
}

define <2 x i32> @vashri32(<2 x i32>* %A) nounwind {
;CHECK-LABEL: vashri32:
;CHECK: vshr.s32
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = ashr <2 x i32> %tmp1, < i32 31, i32 31 >
	ret <2 x i32> %tmp2
}

define <1 x i64> @vashri64(<1 x i64>* %A) nounwind {
;CHECK-LABEL: vashri64:
;CHECK: vshr.s64
	%tmp1 = load <1 x i64>, <1 x i64>* %A
	%tmp2 = ashr <1 x i64> %tmp1, < i64 63 >
	ret <1 x i64> %tmp2
}

define <16 x i8> @vashrQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: vashrQs8:
;CHECK: vneg.s8
;CHECK: vshl.s8
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = ashr <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vashrQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: vashrQs16:
;CHECK: vneg.s16
;CHECK: vshl.s16
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = ashr <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vashrQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: vashrQs32:
;CHECK: vneg.s32
;CHECK: vshl.s32
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = ashr <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <2 x i64> @vashrQs64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: vashrQs64:
;CHECK: vsub.i64
;CHECK: vshl.s64
	%tmp1 = load <2 x i64>, <2 x i64>* %A
	%tmp2 = load <2 x i64>, <2 x i64>* %B
	%tmp3 = ashr <2 x i64> %tmp1, %tmp2
	ret <2 x i64> %tmp3
}

define <16 x i8> @vashrQi8(<16 x i8>* %A) nounwind {
;CHECK-LABEL: vashrQi8:
;CHECK: vshr.s8
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = ashr <16 x i8> %tmp1, < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >
	ret <16 x i8> %tmp2
}

define <8 x i16> @vashrQi16(<8 x i16>* %A) nounwind {
;CHECK-LABEL: vashrQi16:
;CHECK: vshr.s16
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = ashr <8 x i16> %tmp1, < i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15 >
	ret <8 x i16> %tmp2
}

define <4 x i32> @vashrQi32(<4 x i32>* %A) nounwind {
;CHECK-LABEL: vashrQi32:
;CHECK: vshr.s32
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = ashr <4 x i32> %tmp1, < i32 31, i32 31, i32 31, i32 31 >
	ret <4 x i32> %tmp2
}

define <2 x i64> @vashrQi64(<2 x i64>* %A) nounwind {
;CHECK-LABEL: vashrQi64:
;CHECK: vshr.s64
	%tmp1 = load <2 x i64>, <2 x i64>* %A
	%tmp2 = ashr <2 x i64> %tmp1, < i64 63, i64 63 >
	ret <2 x i64> %tmp2
}
