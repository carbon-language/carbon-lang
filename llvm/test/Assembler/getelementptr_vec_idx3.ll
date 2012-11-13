; RUN: not llvm-as < %s >/dev/null 2> %t
; RUN: FileCheck %s < %t
; Test that vector indices have the same number of elements as the pointer.

; CHECK: getelementptr index type missmatch

define <4 x i32> @test(<4 x i32>* %a) {
  %w = getelementptr <4 x i32>* %a, <2 x i32> <i32 5, i32 9>
  ret i32 %w
}
