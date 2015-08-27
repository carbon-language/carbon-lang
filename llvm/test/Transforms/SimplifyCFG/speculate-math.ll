; RUN: opt -S -simplifycfg -phi-node-folding-threshold=2 < %s | FileCheck %s

declare float @llvm.sqrt.f32(float) nounwind readonly
declare float @llvm.fma.f32(float, float, float) nounwind readonly
declare float @llvm.fmuladd.f32(float, float, float) nounwind readonly
declare float @llvm.fabs.f32(float) nounwind readonly
declare float @llvm.minnum.f32(float, float) nounwind readonly
declare float @llvm.maxnum.f32(float, float) nounwind readonly
declare float @llvm.copysign.f32(float, float) nounwind readonly

; CHECK-LABEL: @sqrt_test(
; CHECK: select
define void @sqrt_test(float addrspace(1)* noalias nocapture %out, float %a) nounwind {
entry:
  %cmp.i = fcmp olt float %a, 0.000000e+00
  br i1 %cmp.i, label %test_sqrt.exit, label %cond.else.i

cond.else.i:                                      ; preds = %entry
  %0 = tail call float @llvm.sqrt.f32(float %a) nounwind readnone
  br label %test_sqrt.exit

test_sqrt.exit:                                   ; preds = %cond.else.i, %entry
  %cond.i = phi float [ %0, %cond.else.i ], [ 0x7FF8000000000000, %entry ]
  store float %cond.i, float addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: @fabs_test(
; CHECK: select
define void @fabs_test(float addrspace(1)* noalias nocapture %out, float %a) nounwind {
entry:
  %cmp.i = fcmp olt float %a, 0.000000e+00
  br i1 %cmp.i, label %test_fabs.exit, label %cond.else.i

cond.else.i:                                      ; preds = %entry
  %0 = tail call float @llvm.fabs.f32(float %a) nounwind readnone
  br label %test_fabs.exit

test_fabs.exit:                                   ; preds = %cond.else.i, %entry
  %cond.i = phi float [ %0, %cond.else.i ], [ 0x7FF8000000000000, %entry ]
  store float %cond.i, float addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: @fma_test(
; CHECK: select
define void @fma_test(float addrspace(1)* noalias nocapture %out, float %a, float %b, float %c) nounwind {
entry:
  %cmp.i = fcmp olt float %a, 0.000000e+00
  br i1 %cmp.i, label %test_fma.exit, label %cond.else.i

cond.else.i:                                      ; preds = %entry
  %0 = tail call float @llvm.fma.f32(float %a, float %b, float %c) nounwind readnone
  br label %test_fma.exit

test_fma.exit:                                   ; preds = %cond.else.i, %entry
  %cond.i = phi float [ %0, %cond.else.i ], [ 0x7FF8000000000000, %entry ]
  store float %cond.i, float addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: @fmuladd_test(
; CHECK: select
define void @fmuladd_test(float addrspace(1)* noalias nocapture %out, float %a, float %b, float %c) nounwind {
entry:
  %cmp.i = fcmp olt float %a, 0.000000e+00
  br i1 %cmp.i, label %test_fmuladd.exit, label %cond.else.i

cond.else.i:                                      ; preds = %entry
  %0 = tail call float @llvm.fmuladd.f32(float %a, float %b, float %c) nounwind readnone
  br label %test_fmuladd.exit

test_fmuladd.exit:                                   ; preds = %cond.else.i, %entry
  %cond.i = phi float [ %0, %cond.else.i ], [ 0x7FF8000000000000, %entry ]
  store float %cond.i, float addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: @minnum_test(
; CHECK: select
define void @minnum_test(float addrspace(1)* noalias nocapture %out, float %a, float %b) nounwind {
entry:
  %cmp.i = fcmp olt float %a, 0.000000e+00
  br i1 %cmp.i, label %test_minnum.exit, label %cond.else.i

cond.else.i:                                      ; preds = %entry
  %0 = tail call float @llvm.minnum.f32(float %a, float %b) nounwind readnone
  br label %test_minnum.exit

test_minnum.exit:                                   ; preds = %cond.else.i, %entry
  %cond.i = phi float [ %0, %cond.else.i ], [ 0x7FF8000000000000, %entry ]
  store float %cond.i, float addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: @maxnum_test(
; CHECK: select
define void @maxnum_test(float addrspace(1)* noalias nocapture %out, float %a, float %b) nounwind {
entry:
  %cmp.i = fcmp olt float %a, 0.000000e+00
  br i1 %cmp.i, label %test_maxnum.exit, label %cond.else.i

cond.else.i:                                      ; preds = %entry
  %0 = tail call float @llvm.maxnum.f32(float %a, float %b) nounwind readnone
  br label %test_maxnum.exit

test_maxnum.exit:                                   ; preds = %cond.else.i, %entry
  %cond.i = phi float [ %0, %cond.else.i ], [ 0x7FF8000000000000, %entry ]
  store float %cond.i, float addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: @copysign_test(
; CHECK: select
define void @copysign_test(float addrspace(1)* noalias nocapture %out, float %a, float %b) nounwind {
entry:
  %cmp.i = fcmp olt float %a, 0.000000e+00
  br i1 %cmp.i, label %test_copysign.exit, label %cond.else.i

cond.else.i:                                      ; preds = %entry
  %0 = tail call float @llvm.copysign.f32(float %a, float %b) nounwind readnone
  br label %test_copysign.exit

test_copysign.exit:                                   ; preds = %cond.else.i, %entry
  %cond.i = phi float [ %0, %cond.else.i ], [ 0x7FF8000000000000, %entry ]
  store float %cond.i, float addrspace(1)* %out, align 4
  ret void
}
