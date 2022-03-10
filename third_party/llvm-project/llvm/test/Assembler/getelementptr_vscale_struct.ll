; RUN: not llvm-as < %s >/dev/null 2> %t
; RUN: FileCheck %s < %t
; Test that a scalable vector struct index is rejected.

; CHECK: invalid getelementptr indices

define <vscale x 2 x i32*> @test7(<vscale x 2 x {i32, i32}*> %a) {
  %w = getelementptr {i32, i32}, <vscale x 2 x {i32, i32}*> %a, <vscale x 2 x i32> zeroinitializer, <vscale x 2 x i32> zeroinitializer
  ret <vscale x 2 x i32*> %w
}
