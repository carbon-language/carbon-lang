; RUN: opt -vector-library=MASSV -inject-tli-mappings -loop-vectorize -force-vector-interleave=1 -mattr=-altivec -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64" 
target triple = "powerpc64le-unknown-linux-gnu"

declare double @cbrt(double) #0
declare float @cbrtf(float) #0

declare double @atanh(double) #0
declare float @atanhf(float) #0

; MASSV is unsupported for AltiVec.
; Check that massv entries are not generated.
define void @cbrt_f64(double* nocapture %varray) {
; CHECK-LABEL: @cbrt_f64(
; CHECK-NOT: __cbrtd2{{.*}}<2 x double>
; CHECK: ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @cbrt(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @cbrt_f32(float* nocapture %varray) {
; CHECK-LABEL: @cbrt_f32(
; CHECK-NOT: __cbrtf4{{.*}}<4 x float>
; CHECK: ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @cbrtf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @atanh_f64(double* nocapture %varray) {
; CHECK-LABEL: @atanh_f64(
; CHECK-NOT: __atanhd2{{.*}}<2 x double>
; CHECK: ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @atanh(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @atanh_f32(float* nocapture %varray) {
; CHECK-LABEL: @atanh_f32(
; CHECK-NOT: __atanhf4{{.*}}<2 x double>
; CHECK: ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @atanhf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

attributes #0 = { nounwind }
