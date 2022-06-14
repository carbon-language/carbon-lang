; RUN: llc %s -o - -fast-isel=true -O0 -verify-machineinstrs | FileCheck %s

target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7-apple-ios8.0.0"

; This test ensures that when fast-isel rewrites uses of the vreg for %tmp29, it also
; updates kill flags on the shift instruction generated as part of the gep.
; This was failing instruction verification.

; CHECK-LABEL: @test

%struct.node = type opaque

declare void @foo([4 x i32], %struct.node*)

define void @test([4 x i32] %xpic.coerce, %struct.node* %t) {
bb:
  %tmp29 = extractvalue [4 x i32] %xpic.coerce, 0
  %tmp40 = bitcast %struct.node* %t to [8 x %struct.node*]*
  %tmp41 = getelementptr inbounds [8 x %struct.node*], [8 x %struct.node*]* %tmp40, i32 0, i32 %tmp29
  %tmp42 = load %struct.node*, %struct.node** %tmp41, align 4
  call void @foo([4 x i32] %xpic.coerce, %struct.node* %tmp42)
  ret void
}
