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

define i64 @test2(i8 %A, i8 %B) nounwind {
; CHECK: test2:
; CHECK: shrq $4
; CHECK-NOT: movq
; CHECK-NOT: orq
; CHECK: leaq
; CHECK: ret
  %C = zext i8 %A to i64                          ; <i64> [#uses=1]
  %D = shl i64 %C, 4                              ; <i64> [#uses=1]
  %E = and i64 %D, 48                             ; <i64> [#uses=1]
  %F = zext i8 %B to i64                          ; <i64> [#uses=1]
  %G = lshr i64 %F, 4                             ; <i64> [#uses=1]
  %H = or i64 %G, %E                              ; <i64> [#uses=1]
  ret i64 %H
}
