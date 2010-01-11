; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
; rdar://7527734

define i32 @test(i32 %x) nounwind readnone ssp {
entry:
; CHECK: test:
; CHECK: leal 3(%rdi), %eax
  %0 = shl i32 %x, 5                              ; <i32> [#uses=1]
  %1 = or i32 %0, 3                               ; <i32> [#uses=1]
  ret i32 %1
}
