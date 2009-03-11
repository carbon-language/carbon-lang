; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah -stack-alignment=16 -o %t -f
; RUN: grep pextrw %t | count 2
; RUN: grep pinsrw %t | count 4
; RUN: grep orw %t | count 1
; RUN: grep andw %t | count 1

; Test yonah where we convert a shuffle to pextrw and pinrsw
define <16 x i8> @shuf1(<16 x i8> %T0) nounwind readnone {
entry:
	%tmp8 = shufflevector <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 1, i8 1, i8 1, i8 1, i8 0, i8 0, i8 0, i8 0,  i8 0, i8 0, i8 0, i8 0>, <16 x i8> %T0, <16 x i32> < i32 0, i32 1, i32 16, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef , i32 undef >
	%tmp9 = shufflevector <16 x i8> %tmp8, <16 x i8> %T0,  <16 x i32> < i32 0, i32 1, i32 2, i32 17,  i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef , i32 undef >
	ret <16 x i8> %tmp9
}

