; RUN: opt < %s -cost-model -costmodel-reduxcost=true -analyze -mcpu=core2 -mtriple=x86_64-apple-darwin | FileCheck %s

define fastcc float @reduction_cost_float(<4 x float> %rdx) {
  %rdx.shuf = shufflevector <4 x float> %rdx, <4 x float> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx = fadd <4 x float> %rdx, %rdx.shuf
  %rdx.shuf7 = shufflevector <4 x float> %bin.rdx, <4 x float> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx8 = fadd <4 x float> %bin.rdx, %rdx.shuf7

; Check that we recognize the tree starting at the extractelement as a
; reduction.
; CHECK-LABEL: reduction_cost
; CHECK:  cost of 9 {{.*}} extractelement

  %r = extractelement <4 x float> %bin.rdx8, i32 0
  ret float %r
}

define fastcc i32 @reduction_cost_int(<8 x i32> %rdx) {
  %rdx.shuf = shufflevector <8 x i32> %rdx, <8 x i32> undef,
   <8 x i32> <i32 4    , i32     5, i32     6, i32     7,
              i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx = add <8 x i32> %rdx, %rdx.shuf
  %rdx.shuf.2 = shufflevector <8 x i32> %bin.rdx, <8 x i32> undef,
   <8 x i32> <i32 2    , i32 3,     i32 undef, i32 undef,
              i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx.2 = add <8 x i32> %bin.rdx, %rdx.shuf.2
  %rdx.shuf.3 = shufflevector <8 x i32> %bin.rdx.2, <8 x i32> undef,
   <8 x i32> <i32 1    , i32 undef, i32 undef, i32 undef,
              i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx.3 = add <8 x i32> %bin.rdx.2, %rdx.shuf.3

; CHECK-LABEL: reduction_cost_int
; CHECK:  cost of 23 {{.*}} extractelement

  %r = extractelement <8 x i32> %bin.rdx.3, i32 0
  ret i32 %r
}

define fastcc float @pairwise_hadd(<4 x float> %rdx, float %f1) {
  %rdx.shuf.0.0 = shufflevector <4 x float> %rdx, <4 x float> undef,
        <4 x i32> <i32 0, i32 2 , i32 undef, i32 undef>
  %rdx.shuf.0.1 = shufflevector <4 x float> %rdx, <4 x float> undef,
        <4 x i32> <i32 1, i32 3, i32 undef, i32 undef>
  %bin.rdx.0 = fadd <4 x float> %rdx.shuf.0.0, %rdx.shuf.0.1
  %rdx.shuf.1.0 = shufflevector <4 x float> %bin.rdx.0, <4 x float> undef,
        <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  %rdx.shuf.1.1 = shufflevector <4 x float> %bin.rdx.0, <4 x float> undef,
        <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx.1 = fadd <4 x float> %rdx.shuf.1.0, %rdx.shuf.1.1

; CHECK-LABEL: pairwise_hadd
; CHECK: cost of 11 {{.*}} extractelement

  %r = extractelement <4 x float> %bin.rdx.1, i32 0
  %r2 = fadd float %r, %f1
  ret float %r2
}
define fastcc float @pairwise_hadd_assoc(<4 x float> %rdx, float %f1) {
  %rdx.shuf.0.0 = shufflevector <4 x float> %rdx, <4 x float> undef,
        <4 x i32> <i32 0, i32 2 , i32 undef, i32 undef>
  %rdx.shuf.0.1 = shufflevector <4 x float> %rdx, <4 x float> undef,
        <4 x i32> <i32 1, i32 3, i32 undef, i32 undef>
  %bin.rdx.0 = fadd <4 x float> %rdx.shuf.0.1, %rdx.shuf.0.0
  %rdx.shuf.1.0 = shufflevector <4 x float> %bin.rdx.0, <4 x float> undef,
        <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  %rdx.shuf.1.1 = shufflevector <4 x float> %bin.rdx.0, <4 x float> undef,
        <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx.1 = fadd <4 x float> %rdx.shuf.1.0, %rdx.shuf.1.1

; CHECK-LABEL: pairwise_hadd_assoc
; CHECK: cost of 11 {{.*}} extractelement

  %r = extractelement <4 x float> %bin.rdx.1, i32 0
  %r2 = fadd float %r, %f1
  ret float %r2
}

define fastcc float @pairwise_hadd_skip_first(<4 x float> %rdx, float %f1) {
  %rdx.shuf.0.0 = shufflevector <4 x float> %rdx, <4 x float> undef,
        <4 x i32> <i32 0, i32 2 , i32 undef, i32 undef>
  %rdx.shuf.0.1 = shufflevector <4 x float> %rdx, <4 x float> undef,
        <4 x i32> <i32 1, i32 3, i32 undef, i32 undef>
  %bin.rdx.0 = fadd <4 x float> %rdx.shuf.0.0, %rdx.shuf.0.1
  %rdx.shuf.1.1 = shufflevector <4 x float> %bin.rdx.0, <4 x float> undef,
        <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx.1 = fadd <4 x float> %bin.rdx.0, %rdx.shuf.1.1

; CHECK-LABEL: pairwise_hadd_skip_first
; CHECK: cost of 11 {{.*}} extractelement

  %r = extractelement <4 x float> %bin.rdx.1, i32 0
  %r2 = fadd float %r, %f1
  ret float %r2
}
