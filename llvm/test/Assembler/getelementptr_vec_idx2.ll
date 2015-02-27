; RUN: not llvm-as < %s >/dev/null 2> %t
; RUN: FileCheck %s < %t
; Test that a vector pointer is only used with a vector index.

; CHECK: getelementptr index type missmatch

define <2 x i32> @test(<2 x i32*> %a) {
  %w = getelementptr i32, <2 x i32*> %a, i32 2
  ret <2 x i32> %w
}
