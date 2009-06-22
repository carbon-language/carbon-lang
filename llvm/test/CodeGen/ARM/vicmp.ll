; RUN: llvm-as < %s | llc -march=arm -mattr=+neon > %t
; RUN: grep {vceq\\.i8} %t | count 2
; RUN: grep {vceq\\.i16} %t | count 2
; RUN: grep {vceq\\.i32} %t | count 2
; RUN: grep vmvn %t | count 6
; RUN: grep {vcgt\\.s8} %t | count 1
; RUN: grep {vcge\\.s16} %t | count 1
; RUN: grep {vcgt\\.u16} %t | count 1
; RUN: grep {vcge\\.u32} %t | count 1

; This tests vicmp operations that do not map directly to NEON instructions.
; Not-equal (ne) operations are implemented by VCEQ/VMVN.  Less-than (lt/ult)
; and less-than-or-equal (le/ule) are implemented by swapping the arguments
; to VCGT and VCGE.  Test all the operand types for not-equal but only sample
; the other operations.

define <8 x i8> @vcnei8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = vicmp ne <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vcnei16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = vicmp ne <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vcnei32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = vicmp ne <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <16 x i8> @vcneQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = vicmp ne <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vcneQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = vicmp ne <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vcneQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = vicmp ne <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <16 x i8> @vcltQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = vicmp slt <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <4 x i16> @vcles16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = vicmp sle <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <4 x i16> @vcltu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = vicmp ult <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <4 x i32> @vcleQu32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = vicmp ule <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}
