; RUN: opt < %s -basicaa -slp-vectorizer -S | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

@A = common global [2000 x double] zeroinitializer, align 16
@B = common global [2000 x double] zeroinitializer, align 16
@C = common global [2000 x float] zeroinitializer, align 16
@D = common global [2000 x float] zeroinitializer, align 16

; Currently SCEV isn't smart enough to figure out that accesses
; A[3*i], A[3*i+1] and A[3*i+2] are consecutive, but in future
; that would hopefully be fixed. For now, check that this isn't
; vectorized.
; CHECK-LABEL: foo_3double
; CHECK-NOT: x double>
; Function Attrs: nounwind ssp uwtable
define void @foo_3double(i32 %u) #0 {
entry:
  %u.addr = alloca i32, align 4
  store i32 %u, i32* %u.addr, align 4
  %mul = mul nsw i32 %u, 3
  %idxprom = sext i32 %mul to i64
  %arrayidx = getelementptr inbounds [2000 x double]* @A, i32 0, i64 %idxprom
  %0 = load double* %arrayidx, align 8
  %arrayidx4 = getelementptr inbounds [2000 x double]* @B, i32 0, i64 %idxprom
  %1 = load double* %arrayidx4, align 8
  %add5 = fadd double %0, %1
  store double %add5, double* %arrayidx, align 8
  %add11 = add nsw i32 %mul, 1
  %idxprom12 = sext i32 %add11 to i64
  %arrayidx13 = getelementptr inbounds [2000 x double]* @A, i32 0, i64 %idxprom12
  %2 = load double* %arrayidx13, align 8
  %arrayidx17 = getelementptr inbounds [2000 x double]* @B, i32 0, i64 %idxprom12
  %3 = load double* %arrayidx17, align 8
  %add18 = fadd double %2, %3
  store double %add18, double* %arrayidx13, align 8
  %add24 = add nsw i32 %mul, 2
  %idxprom25 = sext i32 %add24 to i64
  %arrayidx26 = getelementptr inbounds [2000 x double]* @A, i32 0, i64 %idxprom25
  %4 = load double* %arrayidx26, align 8
  %arrayidx30 = getelementptr inbounds [2000 x double]* @B, i32 0, i64 %idxprom25
  %5 = load double* %arrayidx30, align 8
  %add31 = fadd double %4, %5
  store double %add31, double* %arrayidx26, align 8
  ret void
}

; SCEV should be able to tell that accesses A[C1 + C2*i], A[C1 + C2*i], ...
; A[C1 + C2*i] are consecutive, if C2 is a power of 2, and C2 > C1 > 0.
; Thus, the following code should be vectorized.
; CHECK-LABEL: foo_2double
; CHECK: x double>
; Function Attrs: nounwind ssp uwtable
define void @foo_2double(i32 %u) #0 {
entry:
  %u.addr = alloca i32, align 4
  store i32 %u, i32* %u.addr, align 4
  %mul = mul nsw i32 %u, 2
  %idxprom = sext i32 %mul to i64
  %arrayidx = getelementptr inbounds [2000 x double]* @A, i32 0, i64 %idxprom
  %0 = load double* %arrayidx, align 8
  %arrayidx4 = getelementptr inbounds [2000 x double]* @B, i32 0, i64 %idxprom
  %1 = load double* %arrayidx4, align 8
  %add5 = fadd double %0, %1
  store double %add5, double* %arrayidx, align 8
  %add11 = add nsw i32 %mul, 1
  %idxprom12 = sext i32 %add11 to i64
  %arrayidx13 = getelementptr inbounds [2000 x double]* @A, i32 0, i64 %idxprom12
  %2 = load double* %arrayidx13, align 8
  %arrayidx17 = getelementptr inbounds [2000 x double]* @B, i32 0, i64 %idxprom12
  %3 = load double* %arrayidx17, align 8
  %add18 = fadd double %2, %3
  store double %add18, double* %arrayidx13, align 8
  ret void
}

; Similar to the previous test, but with different datatype.
; CHECK-LABEL: foo_4float
; CHECK: x float>
; Function Attrs: nounwind ssp uwtable
define void @foo_4float(i32 %u) #0 {
entry:
  %u.addr = alloca i32, align 4
  store i32 %u, i32* %u.addr, align 4
  %mul = mul nsw i32 %u, 4
  %idxprom = sext i32 %mul to i64
  %arrayidx = getelementptr inbounds [2000 x float]* @C, i32 0, i64 %idxprom
  %0 = load float* %arrayidx, align 4
  %arrayidx4 = getelementptr inbounds [2000 x float]* @D, i32 0, i64 %idxprom
  %1 = load float* %arrayidx4, align 4
  %add5 = fadd float %0, %1
  store float %add5, float* %arrayidx, align 4
  %add11 = add nsw i32 %mul, 1
  %idxprom12 = sext i32 %add11 to i64
  %arrayidx13 = getelementptr inbounds [2000 x float]* @C, i32 0, i64 %idxprom12
  %2 = load float* %arrayidx13, align 4
  %arrayidx17 = getelementptr inbounds [2000 x float]* @D, i32 0, i64 %idxprom12
  %3 = load float* %arrayidx17, align 4
  %add18 = fadd float %2, %3
  store float %add18, float* %arrayidx13, align 4
  %add24 = add nsw i32 %mul, 2
  %idxprom25 = sext i32 %add24 to i64
  %arrayidx26 = getelementptr inbounds [2000 x float]* @C, i32 0, i64 %idxprom25
  %4 = load float* %arrayidx26, align 4
  %arrayidx30 = getelementptr inbounds [2000 x float]* @D, i32 0, i64 %idxprom25
  %5 = load float* %arrayidx30, align 4
  %add31 = fadd float %4, %5
  store float %add31, float* %arrayidx26, align 4
  %add37 = add nsw i32 %mul, 3
  %idxprom38 = sext i32 %add37 to i64
  %arrayidx39 = getelementptr inbounds [2000 x float]* @C, i32 0, i64 %idxprom38
  %6 = load float* %arrayidx39, align 4
  %arrayidx43 = getelementptr inbounds [2000 x float]* @D, i32 0, i64 %idxprom38
  %7 = load float* %arrayidx43, align 4
  %add44 = fadd float %6, %7
  store float %add44, float* %arrayidx39, align 4
  ret void
}

; Similar to the previous tests, but now we are dealing with AddRec SCEV.
; CHECK-LABEL: foo_loop
; CHECK: x double>
; Function Attrs: nounwind ssp uwtable
define i32 @foo_loop(double* %A, i32 %n) #0 {
entry:
  %A.addr = alloca double*, align 8
  %n.addr = alloca i32, align 4
  %sum = alloca double, align 8
  %i = alloca i32, align 4
  store double* %A, double** %A.addr, align 8
  store i32 %n, i32* %n.addr, align 4
  store double 0.000000e+00, double* %sum, align 8
  store i32 0, i32* %i, align 4
  %cmp1 = icmp slt i32 0, %n
  br i1 %cmp1, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %0 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %1 = phi double [ 0.000000e+00, %for.body.lr.ph ], [ %add7, %for.body ]
  %mul = mul nsw i32 %0, 2
  %idxprom = sext i32 %mul to i64
  %arrayidx = getelementptr inbounds double* %A, i64 %idxprom
  %2 = load double* %arrayidx, align 8
  %mul1 = fmul double 7.000000e+00, %2
  %add = add nsw i32 %mul, 1
  %idxprom3 = sext i32 %add to i64
  %arrayidx4 = getelementptr inbounds double* %A, i64 %idxprom3
  %3 = load double* %arrayidx4, align 8
  %mul5 = fmul double 7.000000e+00, %3
  %add6 = fadd double %mul1, %mul5
  %add7 = fadd double %1, %add6
  store double %add7, double* %sum, align 8
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %i, align 4
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  %split = phi double [ %add7, %for.body ]
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  %.lcssa = phi double [ %split, %for.cond.for.end_crit_edge ], [ 0.000000e+00, %entry ]
  %conv = fptosi double %.lcssa to i32
  ret i32 %conv
}

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = metadata !{metadata !"clang version 3.5.0 "}
