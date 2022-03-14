; RUN: not llvm-as < %s >/dev/null 2> %t
; RUN: FileCheck %s < %t
; Test that a vector GEP may be used with a scalar base, the result is a vector of pointers

; CHECK: '%w' defined with type '<2 x <4 x i32>*>'

define <4 x i32> @test(<4 x i32>* %a) {
  %w = getelementptr <4 x i32>, <4 x i32>* %a, <2 x i32> <i32 5, i32 9>
  ret i32 %w
}
