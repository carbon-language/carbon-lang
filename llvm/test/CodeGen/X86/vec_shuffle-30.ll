; RUN: llvm-as < %s | llc -march=x86 -mattr=sse41 -disable-mmx -o %t -f
; RUN: grep pshufhw %t | grep 161 | count 1
; RUN: grep shufps %t | count 1
; RUN: not grep pslldq %t

; Test case when creating pshufhw, we incorrectly set the higher order bit
; for an undef,
define void @test(<8 x i16>* %dest, <8 x i16> %in) {
entry:
  %0 = load <8 x i16>* %dest
  %1 = shufflevector <8 x i16> %0, <8 x i16> %in, <8 x i32> < i32 0, i32 1, i32 2, i32 3, i32 13, i32 undef, i32 14, i32 14>
  store <8 x i16> %1, <8 x i16>* %dest
  ret void
}                              

; A test case where we shouldn't generate a punpckldq but a pshufd and a pslldq
define void @test2(<4 x i32>* %dest, <4 x i32> %in) {
entry:
  %0 = shufflevector <4 x i32> %in, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, <4 x i32> < i32 undef, i32 5, i32 undef, i32 2>
  store <4 x i32> %0, <4 x i32>* %dest
  ret void
}
