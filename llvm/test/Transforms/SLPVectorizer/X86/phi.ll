; RUN: opt < %s -basicaa -slp-vectorizer -slp-threshold=-100 -dce -S -mtriple=i386-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.9.0"

;int foo(double *A, int k) {
;  double A0;
;  double A1;
;  if (k) {
;    A0 = 3;
;    A1 = 5;
;  } else {
;    A0 = A[10];
;    A1 = A[11];
;  }
;  A[0] = A0;
;  A[1] = A1;
;}


;CHECK: i32 @foo
;CHECK: load <2 x double>
;CHECK: phi <2 x double>
;CHECK: store <2 x double>
;CHECK: ret i32 undef
define i32 @foo(double* nocapture %A, i32 %k) {
entry:
  %tobool = icmp eq i32 %k, 0
  br i1 %tobool, label %if.else, label %if.end

if.else:                                          ; preds = %entry
  %arrayidx = getelementptr inbounds double, double* %A, i64 10
  %0 = load double* %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds double, double* %A, i64 11
  %1 = load double* %arrayidx1, align 8
  br label %if.end

if.end:                                           ; preds = %entry, %if.else
  %A0.0 = phi double [ %0, %if.else ], [ 3.000000e+00, %entry ]
  %A1.0 = phi double [ %1, %if.else ], [ 5.000000e+00, %entry ]
  store double %A0.0, double* %A, align 8
  %arrayidx3 = getelementptr inbounds double, double* %A, i64 1
  store double %A1.0, double* %arrayidx3, align 8
  ret i32 undef
}


;int foo(double * restrict B,  double * restrict A, int n, int m) {
;  double R=A[1];
;  double G=A[0];
;  for (int i=0; i < 100; i++) {
;    R += 10;
;    G += 10;
;    R *= 4;
;    G *= 4;
;    R += 4;
;    G += 4;
;  }
;  B[0] = G;
;  B[1] = R;
;  return 0;
;}

;CHECK: foo2
;CHECK: load <2 x double>
;CHECK: phi <2 x double>
;CHECK: fmul <2 x double>
;CHECK: store <2 x double>
;CHECK: ret
define i32 @foo2(double* noalias nocapture %B, double* noalias nocapture %A, i32 %n, i32 %m) #0 {
entry:
  %arrayidx = getelementptr inbounds double, double* %A, i64 1
  %0 = load double* %arrayidx, align 8
  %1 = load double* %A, align 8
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.019 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %G.018 = phi double [ %1, %entry ], [ %add5, %for.body ]
  %R.017 = phi double [ %0, %entry ], [ %add4, %for.body ]
  %add = fadd double %R.017, 1.000000e+01
  %add2 = fadd double %G.018, 1.000000e+01
  %mul = fmul double %add, 4.000000e+00
  %mul3 = fmul double %add2, 4.000000e+00
  %add4 = fadd double %mul, 4.000000e+00
  %add5 = fadd double %mul3, 4.000000e+00
  %inc = add nsw i32 %i.019, 1
  %exitcond = icmp eq i32 %inc, 100
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  store double %add5, double* %B, align 8
  %arrayidx7 = getelementptr inbounds double, double* %B, i64 1
  store double %add4, double* %arrayidx7, align 8
  ret i32 0
}

; float foo3(float *A) {
;
;   float R = A[0];
;   float G = A[1];
;   float B = A[2];
;   float Y = A[3];
;   float P = A[4];
;   for (int i=0; i < 121; i+=3) {
;     R+=A[i+0]*7;
;     G+=A[i+1]*8;
;     B+=A[i+2]*9;
;     Y+=A[i+3]*10;
;     P+=A[i+4]*11;
;   }
;
;   return R+G+B+Y+P;
; }

;CHECK: foo3
;CHECK: phi <4 x float>
;CHECK: fmul <4 x float>
;CHECK: fadd <4 x float>
;CHECK-NOT: phi <5 x float>
;CHECK-NOT: fmul <5 x float>
;CHECK-NOT: fadd <5 x float>

define float @foo3(float* nocapture readonly %A) #0 {
entry:
  %0 = load float* %A, align 4
  %arrayidx1 = getelementptr inbounds float, float* %A, i64 1
  %1 = load float* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds float, float* %A, i64 2
  %2 = load float* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds float, float* %A, i64 3
  %3 = load float* %arrayidx3, align 4
  %arrayidx4 = getelementptr inbounds float, float* %A, i64 4
  %4 = load float* %arrayidx4, align 4
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %P.056 = phi float [ %4, %entry ], [ %add26, %for.body ]
  %Y.055 = phi float [ %3, %entry ], [ %add21, %for.body ]
  %B.054 = phi float [ %2, %entry ], [ %add16, %for.body ]
  %G.053 = phi float [ %1, %entry ], [ %add11, %for.body ]
  %R.052 = phi float [ %0, %entry ], [ %add6, %for.body ]
  %5 = phi float [ %1, %entry ], [ %11, %for.body ]
  %6 = phi float [ %0, %entry ], [ %9, %for.body ]
  %mul = fmul float %6, 7.000000e+00
  %add6 = fadd float %R.052, %mul
  %mul10 = fmul float %5, 8.000000e+00
  %add11 = fadd float %G.053, %mul10
  %7 = add nsw i64 %indvars.iv, 2
  %arrayidx14 = getelementptr inbounds float, float* %A, i64 %7
  %8 = load float* %arrayidx14, align 4
  %mul15 = fmul float %8, 9.000000e+00
  %add16 = fadd float %B.054, %mul15
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 3
  %arrayidx19 = getelementptr inbounds float, float* %A, i64 %indvars.iv.next
  %9 = load float* %arrayidx19, align 4
  %mul20 = fmul float %9, 1.000000e+01
  %add21 = fadd float %Y.055, %mul20
  %10 = add nsw i64 %indvars.iv, 4
  %arrayidx24 = getelementptr inbounds float, float* %A, i64 %10
  %11 = load float* %arrayidx24, align 4
  %mul25 = fmul float %11, 1.100000e+01
  %add26 = fadd float %P.056, %mul25
  %12 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %12, 121
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %add28 = fadd float %add6, %add11
  %add29 = fadd float %add28, %add16
  %add30 = fadd float %add29, %add21
  %add31 = fadd float %add30, %add26
  ret float %add31
}

; Make sure the order of phi nodes of different types does not prevent
; vectorization of same typed phi nodes.
; CHECK-LABEL: sort_phi_type
; CHECK: phi <4 x float>
; CHECK: fmul <4 x float>

define float @sort_phi_type(float* nocapture readonly %A) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %Y = phi float [ 1.000000e+01, %entry ], [ %mul10, %for.body ]
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %B = phi float [ 1.000000e+01, %entry ], [ %mul15, %for.body ]
  %G = phi float [ 1.000000e+01, %entry ], [ %mul20, %for.body ]
  %R = phi float [ 1.000000e+01, %entry ], [ %mul25, %for.body ]
  %mul10 = fmul float %Y, 8.000000e+00
  %mul15 = fmul float %B, 9.000000e+00
  %mul20 = fmul float %R, 10.000000e+01
  %mul25 = fmul float %G, 11.100000e+01
  %indvars.iv.next = add nsw i64 %indvars.iv, 4
  %cmp = icmp slt i64 %indvars.iv.next, 128
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %add28 = fadd float 1.000000e+01, %mul10
  %add29 = fadd float %mul10, %mul15
  %add30 = fadd float %add29, %mul20
  %add31 = fadd float %add30, %mul25
  ret float %add31
}

define void @test(x86_fp80* %i1, x86_fp80* %i2, x86_fp80* %o) {
; CHECK-LABEL: @test(
;
; Test that we correctly recognize the discontiguous memory in arrays where the
; size is less than the alignment, and through various different GEP formations.
;
; We disable the vectorization of x86_fp80 for now. 

entry:
  %i1.0 = load x86_fp80* %i1, align 16
  %i1.gep1 = getelementptr x86_fp80, x86_fp80* %i1, i64 1
  %i1.1 = load x86_fp80* %i1.gep1, align 16
; CHECK: load x86_fp80*
; CHECK: load x86_fp80*
; CHECK-NOT: insertelement <2 x x86_fp80>
; CHECK-NOT: insertelement <2 x x86_fp80>
  br i1 undef, label %then, label %end

then:
  %i2.gep0 = getelementptr inbounds x86_fp80, x86_fp80* %i2, i64 0
  %i2.0 = load x86_fp80* %i2.gep0, align 16
  %i2.gep1 = getelementptr inbounds x86_fp80, x86_fp80* %i2, i64 1
  %i2.1 = load x86_fp80* %i2.gep1, align 16
; CHECK: load x86_fp80*
; CHECK: load x86_fp80*
; CHECK-NOT: insertelement <2 x x86_fp80>
; CHECK-NOT: insertelement <2 x x86_fp80>
  br label %end

end:
  %phi0 = phi x86_fp80 [ %i1.0, %entry ], [ %i2.0, %then ]
  %phi1 = phi x86_fp80 [ %i1.1, %entry ], [ %i2.1, %then ]
; CHECK-NOT: phi <2 x x86_fp80>
; CHECK-NOT: extractelement <2 x x86_fp80>
; CHECK-NOT: extractelement <2 x x86_fp80>
  store x86_fp80 %phi0, x86_fp80* %o, align 16
  %o.gep1 = getelementptr inbounds x86_fp80, x86_fp80* %o, i64 1
  store x86_fp80 %phi1, x86_fp80* %o.gep1, align 16
  ret void
}
