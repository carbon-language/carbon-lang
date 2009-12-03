; RUN: llc -mcpu=cortex-a8 < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "armv7-apple-darwin10"

%struct.int16x8_t = type { <8 x i16> }
%struct.int16x8x2_t = type { [2 x %struct.int16x8_t] }

define arm_apcscc void @t(%struct.int16x8x2_t* noalias nocapture sret %agg.result, <8 x i16> %tmp.0, %struct.int16x8x2_t* nocapture %dst) nounwind {
entry:
;CHECK: vtrn.16
  %0 = shufflevector <8 x i16> %tmp.0, <8 x i16> undef, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  %1 = shufflevector <8 x i16> %tmp.0, <8 x i16> undef, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  %agg.result1218.0 = getelementptr %struct.int16x8x2_t* %agg.result, i32 0, i32 0, i32 0, i32 0 ; <<8 x i16>*>
  store <8 x i16> %0, <8 x i16>* %agg.result1218.0, align 16
  %agg.result12.1.0 = getelementptr %struct.int16x8x2_t* %agg.result, i32 0, i32 0, i32 1, i32 0 ; <<8 x i16>*>
  store <8 x i16> %1, <8 x i16>* %agg.result12.1.0, align 16
  ret void
}
