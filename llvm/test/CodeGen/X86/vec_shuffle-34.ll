; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah -o %t -f
; RUN: grep pextrw %t | count 1
; RUN: grep punpcklqdq %t | count 1
; RUN: grep pshuflw %t | count 1
; RUN: grep pinsrw %t | count 1
; RUN: llvm-as < %s | llc -march=x86 -mcpu=core2 -o %t -f
; RUN: grep pshufb %t | count 2

define <8 x i16> @shuf2(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
	%tmp8 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 undef, i32 undef, i32 7, i32 2, i32 8, i32 undef, i32 undef , i32 undef >
	ret <8 x i16> %tmp8
}
