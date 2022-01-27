; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define <4 x i16> @f(<4 x i32> %vqdmlal_v3.i, <8 x i16> %x5) {
entry:
  ; Check that we don't just dup the input vector. The code emitted is ext, dup, ext, ext
  ; but only match the last three instructions as the first two could be combined to
  ; a dup2 at some stage.
  ; CHECK: dup
  ; CHECK: ext
  ; CHECK: ext
  %x4 = extractelement <4 x i32> %vqdmlal_v3.i, i32 2
  %vgetq_lane = trunc i32 %x4 to i16
  %vecinit.i = insertelement <4 x i16> undef, i16 %vgetq_lane, i32 0
  %vecinit2.i = insertelement <4 x i16> %vecinit.i, i16 %vgetq_lane, i32 2
  %vecinit3.i = insertelement <4 x i16> %vecinit2.i, i16 %vgetq_lane, i32 3
  %vgetq_lane261 = extractelement <8 x i16> %x5, i32 0
  %vset_lane267 = insertelement <4 x i16> %vecinit3.i, i16 %vgetq_lane261, i32 1
  ret <4 x i16> %vset_lane267
}
