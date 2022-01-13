; RUN: opt -mcpu=kryo -mtriple=aarch64-gnu-linux -loop-data-prefetch -max-prefetch-iters-ahead=1000 -S < %s | FileCheck %s --check-prefix=LARGE_PREFETCH --check-prefix=ALL
; RUN: opt -mcpu=kryo -mtriple=aarch64-gnu-linux -loop-data-prefetch -S < %s | FileCheck %s --check-prefix=NO_LARGE_PREFETCH --check-prefix=ALL
; RUN: opt -mcpu=kryo -mtriple=aarch64-gnu-linux -passes=loop-data-prefetch -max-prefetch-iters-ahead=1000 -S < %s | FileCheck %s --check-prefix=LARGE_PREFETCH --check-prefix=ALL
; RUN: opt -mcpu=kryo -mtriple=aarch64-gnu-linux -passes=loop-data-prefetch -S < %s | FileCheck %s --check-prefix=NO_LARGE_PREFETCH --check-prefix=ALL

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32:64-S128"

; ALL-LABEL: @small_stride(
define void @small_stride(double* nocapture %a, double* nocapture readonly %b) {
entry:
  br label %for.body

; ALL: for.body:
for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %b, i64 %indvars.iv
; ALL-NOT: call void @llvm.prefetch
  %0 = load double, double* %arrayidx, align 8
  %add = fadd double %0, 1.000000e+00
  %arrayidx2 = getelementptr inbounds double, double* %a, i64 %indvars.iv
  store double %add, double* %arrayidx2, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1600
  br i1 %exitcond, label %for.end, label %for.body

; ALL: for.end:
for.end:                                          ; preds = %for.body
  ret void
}

; ALL-LABEL: @large_stride(
define void @large_stride(double* nocapture %a, double* nocapture readonly %b) {
entry:
  br label %for.body

; ALL: for.body:
for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %b, i64 %indvars.iv
; LARGE_PREFETCH: call void @llvm.prefetch
; NO_LARGE_PREFETCH-NOT: call void @llvm.prefetch
  %0 = load double, double* %arrayidx, align 8
  %add = fadd double %0, 1.000000e+00
  %arrayidx2 = getelementptr inbounds double, double* %a, i64 %indvars.iv
  store double %add, double* %arrayidx2, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 150 
  %exitcond = icmp eq i64 %indvars.iv.next, 160000
  br i1 %exitcond, label %for.end, label %for.body

; ALL: for.end:
for.end:                                          ; preds = %for.body
  ret void
}
