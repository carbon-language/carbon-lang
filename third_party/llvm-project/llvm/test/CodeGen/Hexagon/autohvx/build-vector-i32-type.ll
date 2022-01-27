; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that this doesn't crash.
; CHECK: sfcmp

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

@g0 = global <16 x float> zeroinitializer, align 8
@g1 = global <16 x i32> zeroinitializer, align 8

define void @fred(<16 x i16> %x, <16 x float> %y) #0 {
b0:
  %v1 = load <16 x float>, <16 x float>* @g0, align 8
  %v2 = fcmp olt <16 x float> %y, %v1
  %v3 = select <16 x i1> %v2, <16 x i16> %x, <16 x i16> zeroinitializer
  %v4 = sext <16 x i16> %v3 to <16 x i32>
  store <16 x i32> %v4, <16 x i32>* @g1, align 64
  ret void
}

attributes #0 = { noinline norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length64b,+hvxv60" }
