; RUN: not llvm-as < %s >/dev/null 2> %t
; RUN: FileCheck %s < %t
; Test that a vector index is only used with a vector pointer.

; CHECK: getelementptr index type missmatch

define i32 @test(i32* %a) {
  %w = getelementptr i32* %a, <2 x i32> <i32 5, i32 9>
  ret i32 %w
}
