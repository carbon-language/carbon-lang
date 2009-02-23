; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah -o %t -f
; RUN: grep punpcklqdq %t | count 1
; RUN: grep pextrw %t | count 1
; RUN: grep pshufd %t | count 1
; RUN: grep pinsrw %t | count 1
; RUN: llvm-as < %s | llc -march=x86 -mcpu=core2 -o %t -f
; RUN: grep pshufb %t | count 1

define <8 x i16> @shuf4(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
	%tmp9 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 8, i32 9, i32 undef, i32 undef, i32 11, i32 3, i32 undef , i32 undef >
	ret <8 x i16> %tmp9
}
