; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah -o %t -f
; RUN: grep punpcklqdq %t | count 1
; RUN: grep pshufhw %t | count 1
; RUN: not grep pextrw %t
; RUN: not grep pinsrw %t

define <8 x i16> @shuf5(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
	%tmp9 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 8, i32 9, i32 undef, i32 undef, i32 undef, i32 2, i32 undef , i32 undef >
	ret <8 x i16> %tmp9
}
