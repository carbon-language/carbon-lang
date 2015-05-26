; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: f_fu
; CHECK-NOT: vpblend
; CHECK: vmovdqa32 {{.*}} {%k1}

define void @f_fu(float* %ret, float*  %aa, float %b) {
allocas:
  %ptr_cast_for_load = bitcast float* %aa to <16 x float>*
  %ptr_masked_load.39 = load <16 x float>, <16 x float>* %ptr_cast_for_load, align 4
  %b_load_to_int32 = fptosi float %b to i32
  %b_load_to_int32_broadcast_init = insertelement <16 x i32> undef, i32 %b_load_to_int32, i32 0
  %b_load_to_int32_broadcast = shufflevector <16 x i32> %b_load_to_int32_broadcast_init, <16 x i32> undef, <16 x i32> zeroinitializer
  %b_to_int32 = fptosi float %b to i32
  %b_to_int32_broadcast_init = insertelement <16 x i32> undef, i32 %b_to_int32, i32 0
  %b_to_int32_broadcast = shufflevector <16 x i32> %b_to_int32_broadcast_init, <16 x i32> undef, <16 x i32> zeroinitializer

  %a_load_to_int32 = fptosi <16 x float> %ptr_masked_load.39 to <16 x i32>
  %div_v019_load_ = sdiv <16 x i32> %b_to_int32_broadcast, <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>

  %v1.i = select <16 x i1> <i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true>, <16 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17>, <16 x i32> %a_load_to_int32

  %foo_test = add <16 x i32> %div_v019_load_, %b_load_to_int32_broadcast 


  %add_struct_offset_y_struct_offset33_x = add <16 x i32> %foo_test, %v1.i 

  %val = sitofp <16 x i32> %add_struct_offset_y_struct_offset33_x to <16 x float>
  %ptrcast = bitcast float* %ret to <16 x float>*
  store <16 x float> %val, <16 x float>* %ptrcast, align 4
  ret void
}