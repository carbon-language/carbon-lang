; RUN: opt < %s -loop-vectorize -S 2>&1 | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; This test makes sure we don't duplicate the loop vectorizer's metadata
; while marking them as already vectorized (by setting width = 1), even
; at lower optimization levels, where no extra cleanup is done

define void @_Z3fooPf(float* %a) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float* %a, i64 %indvars.iv
  %p = load float* %arrayidx, align 4
  %mul = fmul float %p, 2.000000e+00
  store float %mul, float* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body
  ret void
}

!0 = metadata !{metadata !0, metadata !1}
!1 = metadata !{metadata !"llvm.loop.vectorize.width", i32 4}
; CHECK-NOT: !{metadata !"llvm.loop.vectorize.width", i32 4}
; CHECK: !{metadata !"llvm.loop.interleave.count", i32 1}
