; RUN: llc < %s -march=x86-64 -mcpu=corei7-avx -mtriple=x86_64-pc-win32 | FileCheck %s

; CHECK: test
; CHECK: vpmovzxwd
; CHECK: vpmovzxwd
define void @test(<4 x i64> %a, <4 x i16>* %buf) {
  %ex1 = extractelement <4 x i64> %a, i32 0
  %ex2 = extractelement <4 x i64> %a, i32 1
  %x1 = bitcast i64 %ex1 to <4 x i16>
  %x2 = bitcast i64 %ex2 to <4 x i16>
  %Sh = shufflevector <4 x i16> %x1, <4 x i16> %x2, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  store <4 x i16> %Sh, <4 x i16>* %buf, align 1
  ret void
}
