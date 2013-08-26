; RUN: opt -S -slp-vectorizer -slp-threshold=-10000 < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-n8:16:32:64-S128"

target triple = "x86_64-apple-macosx10.8.0"

define <4 x float> @simple_select(<4 x float> %a, <4 x float> %b, <4 x i32> %c) #0 {
; CHECK-LABEL: @simple_select(
; CHECK-NEXT: %1 = icmp ne <4 x i32> %c, zeroinitializer
; CHECK-NEXT: select <4 x i1> %1, <4 x float> %a, <4 x float> %b
  %c0 = extractelement <4 x i32> %c, i32 0
  %c1 = extractelement <4 x i32> %c, i32 1
  %c2 = extractelement <4 x i32> %c, i32 2
  %c3 = extractelement <4 x i32> %c, i32 3
  %a0 = extractelement <4 x float> %a, i32 0
  %a1 = extractelement <4 x float> %a, i32 1
  %a2 = extractelement <4 x float> %a, i32 2
  %a3 = extractelement <4 x float> %a, i32 3
  %b0 = extractelement <4 x float> %b, i32 0
  %b1 = extractelement <4 x float> %b, i32 1
  %b2 = extractelement <4 x float> %b, i32 2
  %b3 = extractelement <4 x float> %b, i32 3
  %cmp0 = icmp ne i32 %c0, 0
  %cmp1 = icmp ne i32 %c1, 0
  %cmp2 = icmp ne i32 %c2, 0
  %cmp3 = icmp ne i32 %c3, 0
  %s0 = select i1 %cmp0, float %a0, float %b0
  %s1 = select i1 %cmp1, float %a1, float %b1
  %s2 = select i1 %cmp2, float %a2, float %b2
  %s3 = select i1 %cmp3, float %a3, float %b3
  %ra = insertelement <4 x float> undef, float %s0, i32 0
  %rb = insertelement <4 x float> %ra, float %s1, i32 1
  %rc = insertelement <4 x float> %rb, float %s2, i32 2
  %rd = insertelement <4 x float> %rc, float %s3, i32 3
  ret <4 x float> %rd
}

; Insert in an order different from the vector indices to make sure it
; doesn't matter
define <4 x float> @simple_select_insert_out_of_order(<4 x float> %a, <4 x float> %b, <4 x i32> %c) #0 {
; CHECK-LABEL: @simple_select_insert_out_of_order(
; CHECK-NEXT: %1 = icmp ne <4 x i32> %c, zeroinitializer
; CHECK-NEXT: select <4 x i1> %1, <4 x float> %a, <4 x float> %b
  %c0 = extractelement <4 x i32> %c, i32 0
  %c1 = extractelement <4 x i32> %c, i32 1
  %c2 = extractelement <4 x i32> %c, i32 2
  %c3 = extractelement <4 x i32> %c, i32 3
  %a0 = extractelement <4 x float> %a, i32 0
  %a1 = extractelement <4 x float> %a, i32 1
  %a2 = extractelement <4 x float> %a, i32 2
  %a3 = extractelement <4 x float> %a, i32 3
  %b0 = extractelement <4 x float> %b, i32 0
  %b1 = extractelement <4 x float> %b, i32 1
  %b2 = extractelement <4 x float> %b, i32 2
  %b3 = extractelement <4 x float> %b, i32 3
  %cmp0 = icmp ne i32 %c0, 0
  %cmp1 = icmp ne i32 %c1, 0
  %cmp2 = icmp ne i32 %c2, 0
  %cmp3 = icmp ne i32 %c3, 0
  %s0 = select i1 %cmp0, float %a0, float %b0
  %s1 = select i1 %cmp1, float %a1, float %b1
  %s2 = select i1 %cmp2, float %a2, float %b2
  %s3 = select i1 %cmp3, float %a3, float %b3
  %ra = insertelement <4 x float> undef, float %s0, i32 2
  %rb = insertelement <4 x float> %ra, float %s1, i32 1
  %rc = insertelement <4 x float> %rb, float %s2, i32 0
  %rd = insertelement <4 x float> %rc, float %s3, i32 3
  ret <4 x float> %rd
}

declare void @v4f32_user(<4 x float>) #0
declare void @f32_user(float) #0

; Multiple users of the final constructed vector
define <4 x float> @simple_select_users(<4 x float> %a, <4 x float> %b, <4 x i32> %c) #0 {
; CHECK-LABEL: @simple_select_users(
; CHECK-NEXT: %1 = icmp ne <4 x i32> %c, zeroinitializer
; CHECK-NEXT: select <4 x i1> %1, <4 x float> %a, <4 x float> %b
  %c0 = extractelement <4 x i32> %c, i32 0
  %c1 = extractelement <4 x i32> %c, i32 1
  %c2 = extractelement <4 x i32> %c, i32 2
  %c3 = extractelement <4 x i32> %c, i32 3
  %a0 = extractelement <4 x float> %a, i32 0
  %a1 = extractelement <4 x float> %a, i32 1
  %a2 = extractelement <4 x float> %a, i32 2
  %a3 = extractelement <4 x float> %a, i32 3
  %b0 = extractelement <4 x float> %b, i32 0
  %b1 = extractelement <4 x float> %b, i32 1
  %b2 = extractelement <4 x float> %b, i32 2
  %b3 = extractelement <4 x float> %b, i32 3
  %cmp0 = icmp ne i32 %c0, 0
  %cmp1 = icmp ne i32 %c1, 0
  %cmp2 = icmp ne i32 %c2, 0
  %cmp3 = icmp ne i32 %c3, 0
  %s0 = select i1 %cmp0, float %a0, float %b0
  %s1 = select i1 %cmp1, float %a1, float %b1
  %s2 = select i1 %cmp2, float %a2, float %b2
  %s3 = select i1 %cmp3, float %a3, float %b3
  %ra = insertelement <4 x float> undef, float %s0, i32 0
  %rb = insertelement <4 x float> %ra, float %s1, i32 1
  %rc = insertelement <4 x float> %rb, float %s2, i32 2
  %rd = insertelement <4 x float> %rc, float %s3, i32 3
  call void @v4f32_user(<4 x float> %rd) #0
  ret <4 x float> %rd
}

; Unused insertelement
define <4 x float> @simple_select_no_users(<4 x float> %a, <4 x float> %b, <4 x i32> %c) #0 {
; CHECK-LABEL: @simple_select_no_users(
; CHECK-NOT: icmp ne <4 x i32>
; CHECK-NOT: select <4 x i1>
  %c0 = extractelement <4 x i32> %c, i32 0
  %c1 = extractelement <4 x i32> %c, i32 1
  %c2 = extractelement <4 x i32> %c, i32 2
  %c3 = extractelement <4 x i32> %c, i32 3
  %a0 = extractelement <4 x float> %a, i32 0
  %a1 = extractelement <4 x float> %a, i32 1
  %a2 = extractelement <4 x float> %a, i32 2
  %a3 = extractelement <4 x float> %a, i32 3
  %b0 = extractelement <4 x float> %b, i32 0
  %b1 = extractelement <4 x float> %b, i32 1
  %b2 = extractelement <4 x float> %b, i32 2
  %b3 = extractelement <4 x float> %b, i32 3
  %cmp0 = icmp ne i32 %c0, 0
  %cmp1 = icmp ne i32 %c1, 0
  %cmp2 = icmp ne i32 %c2, 0
  %cmp3 = icmp ne i32 %c3, 0
  %s0 = select i1 %cmp0, float %a0, float %b0
  %s1 = select i1 %cmp1, float %a1, float %b1
  %s2 = select i1 %cmp2, float %a2, float %b2
  %s3 = select i1 %cmp3, float %a3, float %b3
  %ra = insertelement <4 x float> undef, float %s0, i32 0
  %rb = insertelement <4 x float> %ra, float %s1, i32 1
  %rc = insertelement <4 x float> undef, float %s2, i32 2
  %rd = insertelement <4 x float> %rc, float %s3, i32 3
  ret <4 x float> %rd
}

; Make sure infinite loop doesn't happen which I ran into when trying
; to do this backwards this backwards
define <4 x i32> @reconstruct(<4 x i32> %c) #0 {
; CHECK-LABEL: @reconstruct(
  %c0 = extractelement <4 x i32> %c, i32 0
  %c1 = extractelement <4 x i32> %c, i32 1
  %c2 = extractelement <4 x i32> %c, i32 2
  %c3 = extractelement <4 x i32> %c, i32 3
  %ra = insertelement <4 x i32> undef, i32 %c0, i32 0
  %rb = insertelement <4 x i32> %ra, i32 %c1, i32 1
  %rc = insertelement <4 x i32> %rb, i32 %c2, i32 2
  %rd = insertelement <4 x i32> %rc, i32 %c3, i32 3
  ret <4 x i32> %rd
}

define <2 x float> @simple_select_v2(<2 x float> %a, <2 x float> %b, <2 x i32> %c) #0 {
; CHECK-LABEL: @simple_select_v2(
; CHECK: icmp ne <2 x i32>
; CHECK: select <2 x i1>
  %c0 = extractelement <2 x i32> %c, i32 0
  %c1 = extractelement <2 x i32> %c, i32 1
  %a0 = extractelement <2 x float> %a, i32 0
  %a1 = extractelement <2 x float> %a, i32 1
  %b0 = extractelement <2 x float> %b, i32 0
  %b1 = extractelement <2 x float> %b, i32 1
  %cmp0 = icmp ne i32 %c0, 0
  %cmp1 = icmp ne i32 %c1, 0
  %s0 = select i1 %cmp0, float %a0, float %b0
  %s1 = select i1 %cmp1, float %a1, float %b1
  %ra = insertelement <2 x float> undef, float %s0, i32 0
  %rb = insertelement <2 x float> %ra, float %s1, i32 1
  ret <2 x float> %rb
}

; Make sure when we construct partial vectors, we don't keep
; re-visiting the insertelement chains starting with undef
; (low cost threshold needed to force this to happen)
define <4 x float> @simple_select_partial_vector(<4 x float> %a, <4 x float> %b, <4 x i32> %c) #0 {
  %c0 = extractelement <4 x i32> %c, i32 0
  %c1 = extractelement <4 x i32> %c, i32 1
  %a0 = extractelement <4 x float> %a, i32 0
  %a1 = extractelement <4 x float> %a, i32 1
  %b0 = extractelement <4 x float> %b, i32 0
  %b1 = extractelement <4 x float> %b, i32 1
  %1 = insertelement <2 x i32> undef, i32 %c0, i32 0
  %2 = insertelement <2 x i32> %1, i32 %c1, i32 1
  %3 = icmp ne <2 x i32> %2, zeroinitializer
  %4 = insertelement <2 x float> undef, float %a0, i32 0
  %5 = insertelement <2 x float> %4, float %a1, i32 1
  %6 = insertelement <2 x float> undef, float %b0, i32 0
  %7 = insertelement <2 x float> %6, float %b1, i32 1
  %8 = select <2 x i1> %3, <2 x float> %5, <2 x float> %7
  %9 = extractelement <2 x float> %8, i32 0
  %ra = insertelement <4 x float> undef, float %9, i32 0
  %10 = extractelement <2 x float> %8, i32 1
  %rb = insertelement <4 x float> %ra, float %10, i32 1
  ret <4 x float> %rb
}

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
