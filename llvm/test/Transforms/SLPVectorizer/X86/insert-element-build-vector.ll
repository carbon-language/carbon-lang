; RUN: opt -S -slp-vectorizer -slp-threshold=-10000 < %s | FileCheck %s
; RUN: opt -S -slp-vectorizer -slp-threshold=0 < %s | FileCheck %s -check-prefix=ZEROTHRESH
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

; Make sure that vectorization happens even if insertelements operations
; must be rescheduled. The case here is from compiling Julia.
define <4 x float> @reschedule_extract(<4 x float> %a, <4 x float> %b) {
; CHECK-LABEL: @reschedule_extract(
; CHECK: %1 = fadd <4 x float> %a, %b
  %a0 = extractelement <4 x float> %a, i32 0
  %b0 = extractelement <4 x float> %b, i32 0
  %c0 = fadd float %a0, %b0
  %v0 = insertelement <4 x float> undef, float %c0, i32 0
  %a1 = extractelement <4 x float> %a, i32 1
  %b1 = extractelement <4 x float> %b, i32 1
  %c1 = fadd float %a1, %b1
  %v1 = insertelement <4 x float> %v0, float %c1, i32 1
  %a2 = extractelement <4 x float> %a, i32 2
  %b2 = extractelement <4 x float> %b, i32 2
  %c2 = fadd float %a2, %b2
  %v2 = insertelement <4 x float> %v1, float %c2, i32 2
  %a3 = extractelement <4 x float> %a, i32 3
  %b3 = extractelement <4 x float> %b, i32 3
  %c3 = fadd float %a3, %b3
  %v3 = insertelement <4 x float> %v2, float %c3, i32 3
  ret <4 x float> %v3
}

; Check that cost model for vectorization takes credit for
; instructions that are erased.
define <4 x float> @take_credit(<4 x float> %a, <4 x float> %b) {
; ZEROTHRESH-LABEL: @take_credit(
; ZEROTHRESH: %1 = fadd <4 x float> %a, %b
  %a0 = extractelement <4 x float> %a, i32 0
  %b0 = extractelement <4 x float> %b, i32 0
  %c0 = fadd float %a0, %b0
  %a1 = extractelement <4 x float> %a, i32 1
  %b1 = extractelement <4 x float> %b, i32 1
  %c1 = fadd float %a1, %b1
  %a2 = extractelement <4 x float> %a, i32 2
  %b2 = extractelement <4 x float> %b, i32 2
  %c2 = fadd float %a2, %b2
  %a3 = extractelement <4 x float> %a, i32 3
  %b3 = extractelement <4 x float> %b, i32 3
  %c3 = fadd float %a3, %b3
  %v0 = insertelement <4 x float> undef, float %c0, i32 0
  %v1 = insertelement <4 x float> %v0, float %c1, i32 1
  %v2 = insertelement <4 x float> %v1, float %c2, i32 2
  %v3 = insertelement <4 x float> %v2, float %c3, i32 3
  ret <4 x float> %v3
}

; Make sure we handle multiple trees that feed one build vector correctly.
define <4 x double> @multi_tree(double %w, double %x, double %y, double %z) {
entry:
  %t0 = fadd double %w , 0.000000e+00
  %t1 = fadd double %x , 1.000000e+00
  %t2 = fadd double %y , 2.000000e+00
  %t3 = fadd double %z , 3.000000e+00
  %t4 = fmul double %t0, 1.000000e+00
  %i1 = insertelement <4 x double> undef, double %t4, i32 3
  %t5 = fmul double %t1, 1.000000e+00
  %i2 = insertelement <4 x double> %i1, double %t5, i32 2
  %t6 = fmul double %t2, 1.000000e+00
  %i3 = insertelement <4 x double> %i2, double %t6, i32 1
  %t7 = fmul double %t3, 1.000000e+00
  %i4 = insertelement <4 x double> %i3, double %t7, i32 0
  ret <4 x double> %i4
}
; CHECK-LABEL: @multi_tree
; CHECK-DAG:  %[[V0:.+]] = insertelement <2 x double> undef, double %w, i32 0
; CHECK-DAG:  %[[V1:.+]] = insertelement <2 x double> %[[V0]], double %x, i32 1
; CHECK-DAG:  %[[V2:.+]] = fadd <2 x double> %[[V1]], <double 0.000000e+00, double 1.000000e+00>
; CHECK-DAG:  %[[V3:.+]] = insertelement <2 x double> undef, double %y, i32 0
; CHECK-DAG:  %[[V4:.+]] = insertelement <2 x double> %[[V3]], double %z, i32 1
; CHECK-DAG:  %[[V5:.+]] = fadd <2 x double> %[[V4]], <double 2.000000e+00, double 3.000000e+00>
; CHECK-DAG:  %[[V6:.+]] = fmul <2 x double> <double 1.000000e+00, double 1.000000e+00>, %[[V2]]
; CHECK-DAG:  %[[V7:.+]] = extractelement <2 x double> %[[V6]], i32 0
; CHECK-DAG:  %[[I1:.+]] = insertelement <4 x double> undef, double %[[V7]], i32 3
; CHECK-DAG:  %[[V8:.+]] = extractelement <2 x double> %[[V6]], i32 1
; CHECK-DAG:  %[[I2:.+]] = insertelement <4 x double> %[[I1]], double %[[V8]], i32 2
; CHECK-DAG:  %[[V9:.+]] = fmul <2 x double> <double 1.000000e+00, double 1.000000e+00>, %[[V5]]
; CHECK-DAG:  %[[V10:.+]] = extractelement <2 x double> %[[V9]], i32 0
; CHECK-DAG:  %[[I3:.+]] = insertelement <4 x double> %i2, double %[[V10]], i32 1
; CHECK-DAG:  %[[V11:.+]] = extractelement <2 x double> %[[V9]], i32 1
; CHECK-DAG:  %[[I4:.+]] = insertelement <4 x double> %i3, double %[[V11]], i32 0
; CHECK:  ret <4 x double> %[[I4]]

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
