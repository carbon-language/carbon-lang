; RUN: llc < %s -march=arm -mattr=+neon > %t
; RUN: grep {vshl\\.s8} %t | count 2
; RUN: grep {vshl\\.s16} %t | count 2
; RUN: grep {vshl\\.s32} %t | count 2
; RUN: grep {vshl\\.s64} %t | count 2
; RUN: grep {vshl\\.u8} %t | count 4
; RUN: grep {vshl\\.u16} %t | count 4
; RUN: grep {vshl\\.u32} %t | count 4
; RUN: grep {vshl\\.u64} %t | count 4
; RUN: grep {vshl\\.i8} %t | count 2
; RUN: grep {vshl\\.i16} %t | count 2
; RUN: grep {vshl\\.i32} %t | count 2
; RUN: grep {vshl\\.i64} %t | count 2
; RUN: grep {vshr\\.u8} %t | count 2
; RUN: grep {vshr\\.u16} %t | count 2
; RUN: grep {vshr\\.u32} %t | count 2
; RUN: grep {vshr\\.u64} %t | count 2
; RUN: grep {vshr\\.s8} %t | count 2
; RUN: grep {vshr\\.s16} %t | count 2
; RUN: grep {vshr\\.s32} %t | count 2
; RUN: grep {vshr\\.s64} %t | count 2
; RUN: grep {vneg\\.s8} %t | count 4
; RUN: grep {vneg\\.s16} %t | count 4
; RUN: grep {vneg\\.s32} %t | count 4
; RUN: grep {vsub\\.i64} %t | count 4

define <8 x i8> @vshls8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = shl <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vshls16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = shl <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vshls32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = shl <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <1 x i64> @vshls64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = shl <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

define <8 x i8> @vshli8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = shl <8 x i8> %tmp1, < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >
	ret <8 x i8> %tmp2
}

define <4 x i16> @vshli16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = shl <4 x i16> %tmp1, < i16 15, i16 15, i16 15, i16 15 >
	ret <4 x i16> %tmp2
}

define <2 x i32> @vshli32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = shl <2 x i32> %tmp1, < i32 31, i32 31 >
	ret <2 x i32> %tmp2
}

define <1 x i64> @vshli64(<1 x i64>* %A) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = shl <1 x i64> %tmp1, < i64 63 >
	ret <1 x i64> %tmp2
}

define <16 x i8> @vshlQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = shl <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vshlQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = shl <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vshlQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = shl <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <2 x i64> @vshlQs64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = shl <2 x i64> %tmp1, %tmp2
	ret <2 x i64> %tmp3
}

define <16 x i8> @vshlQi8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = shl <16 x i8> %tmp1, < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >
	ret <16 x i8> %tmp2
}

define <8 x i16> @vshlQi16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = shl <8 x i16> %tmp1, < i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15 >
	ret <8 x i16> %tmp2
}

define <4 x i32> @vshlQi32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = shl <4 x i32> %tmp1, < i32 31, i32 31, i32 31, i32 31 >
	ret <4 x i32> %tmp2
}

define <2 x i64> @vshlQi64(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = shl <2 x i64> %tmp1, < i64 63, i64 63 >
	ret <2 x i64> %tmp2
}

define <8 x i8> @vlshru8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = lshr <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vlshru16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = lshr <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vlshru32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = lshr <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <1 x i64> @vlshru64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = lshr <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

define <8 x i8> @vlshri8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = lshr <8 x i8> %tmp1, < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
	ret <8 x i8> %tmp2
}

define <4 x i16> @vlshri16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = lshr <4 x i16> %tmp1, < i16 16, i16 16, i16 16, i16 16 >
	ret <4 x i16> %tmp2
}

define <2 x i32> @vlshri32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = lshr <2 x i32> %tmp1, < i32 32, i32 32 >
	ret <2 x i32> %tmp2
}

define <1 x i64> @vlshri64(<1 x i64>* %A) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = lshr <1 x i64> %tmp1, < i64 64 >
	ret <1 x i64> %tmp2
}

define <16 x i8> @vlshrQu8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = lshr <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vlshrQu16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = lshr <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vlshrQu32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = lshr <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <2 x i64> @vlshrQu64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = lshr <2 x i64> %tmp1, %tmp2
	ret <2 x i64> %tmp3
}

define <16 x i8> @vlshrQi8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = lshr <16 x i8> %tmp1, < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
	ret <16 x i8> %tmp2
}

define <8 x i16> @vlshrQi16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = lshr <8 x i16> %tmp1, < i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16 >
	ret <8 x i16> %tmp2
}

define <4 x i32> @vlshrQi32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = lshr <4 x i32> %tmp1, < i32 32, i32 32, i32 32, i32 32 >
	ret <4 x i32> %tmp2
}

define <2 x i64> @vlshrQi64(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = lshr <2 x i64> %tmp1, < i64 64, i64 64 >
	ret <2 x i64> %tmp2
}

define <8 x i8> @vashrs8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = ashr <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vashrs16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = ashr <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vashrs32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = ashr <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <1 x i64> @vashrs64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = ashr <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

define <8 x i8> @vashri8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = ashr <8 x i8> %tmp1, < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
	ret <8 x i8> %tmp2
}

define <4 x i16> @vashri16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = ashr <4 x i16> %tmp1, < i16 16, i16 16, i16 16, i16 16 >
	ret <4 x i16> %tmp2
}

define <2 x i32> @vashri32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = ashr <2 x i32> %tmp1, < i32 32, i32 32 >
	ret <2 x i32> %tmp2
}

define <1 x i64> @vashri64(<1 x i64>* %A) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = ashr <1 x i64> %tmp1, < i64 64 >
	ret <1 x i64> %tmp2
}

define <16 x i8> @vashrQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = ashr <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vashrQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = ashr <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vashrQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = ashr <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <2 x i64> @vashrQs64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = ashr <2 x i64> %tmp1, %tmp2
	ret <2 x i64> %tmp3
}

define <16 x i8> @vashrQi8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = ashr <16 x i8> %tmp1, < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
	ret <16 x i8> %tmp2
}

define <8 x i16> @vashrQi16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = ashr <8 x i16> %tmp1, < i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16 >
	ret <8 x i16> %tmp2
}

define <4 x i32> @vashrQi32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = ashr <4 x i32> %tmp1, < i32 32, i32 32, i32 32, i32 32 >
	ret <4 x i32> %tmp2
}

define <2 x i64> @vashrQi64(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = ashr <2 x i64> %tmp1, < i64 64, i64 64 >
	ret <2 x i64> %tmp2
}
