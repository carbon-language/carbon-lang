; RUN: llc < %s -O=3 -mtriple=arm64-apple-ios -mcpu=cyclone -enable-unsafe-fp-math | FileCheck %s
define void @foo_2d(double* %src) {
; CHECK-LABEL: %entry
; CHECK: fmul {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
; CHECK: fmadd {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %arrayidx1 = getelementptr inbounds double, double* %src, i64 5
  %arrayidx2 = getelementptr inbounds double, double* %src, i64 11
  %tmp = bitcast double* %arrayidx1 to <2 x double>*
  %tmp1 = load double, double* %arrayidx2, align 8
  %tmp2 = load double, double* %arrayidx1, align 8
  %fmul = fmul fast double %tmp1, %tmp1
  %fmul2 = fmul fast double %tmp2, 0x3F94AFD6A052BF5B
  %fadd = fadd fast double %fmul, %fmul2
  br label %for.body

; CHECK-LABEL: %for.body
; CHECK: fmla.2d {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
; CHECK: fmla.2d {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}[0]
; CHECK: fmla.d {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}[0]
for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx3 = getelementptr inbounds double, double* %src, i64 %indvars.iv.next
  %tmp3 = load double, double* %arrayidx3, align 8
  %add = fadd fast double %tmp3, %tmp3
  %mul = fmul fast double %add, %fadd
  %e1 = insertelement <2 x double> undef, double %add, i32 0
  %e2 = insertelement <2 x double> %e1, double %add, i32 1
  %add2 = fadd fast <2 x double> %e2, <double 3.000000e+00, double -3.000000e+00>
  %e3 = insertelement <2 x double> undef, double %mul, i32 0
  %e4 = insertelement <2 x double> %e3, double %mul, i32 1
  %mul2 = fmul fast <2 x double> %add2,<double 3.000000e+00, double -3.000000e+00>
  %e5 = insertelement <2 x double> undef, double %add, i32 0
  %e6 = insertelement <2 x double> %e5, double %add, i32 1
  %add3 = fadd fast  <2 x double> %mul2, <double 3.000000e+00, double -3.000000e+00>
  %mulx = fmul fast <2 x double> %add2, %e2
  %addx = fadd fast  <2 x double> %mulx, %e4
  %e7 = insertelement <2 x double> undef, double %mul, i32 0
  %e8 = insertelement <2 x double> %e7, double %mul, i32 1
  %e9 = fmul fast <2 x double>  %addx, %add3
  store <2 x double> %e9, <2 x double>* %tmp, align 8
  %e10 = extractelement <2 x double> %add3, i32 0
  %mul3 = fmul fast double %mul, %e10
  %add4 = fadd fast double %mul3, %mul
  store double %add4, double* %arrayidx2, align 8
  %exitcond = icmp eq i64 %indvars.iv.next, 25
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
define void @foo_2s(float* %src) {
entry:
  %arrayidx1 = getelementptr inbounds float, float* %src, i64 5
  %arrayidx2 = getelementptr inbounds float, float* %src, i64 11
  %tmp = bitcast float* %arrayidx1 to <2 x float>*
  br label %for.body

; CHECK-LABEL: %for.body
; CHECK: fmla.2s {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
; CHECK: fmla.2s {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}[0]
; CHECK: fmla.s {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}[0]
for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx3 = getelementptr inbounds float, float* %src, i64 %indvars.iv.next
  %tmp1 = load float, float* %arrayidx3, align 8
  %add = fadd fast float %tmp1, %tmp1
  %mul = fmul fast float %add, %add
  %e1 = insertelement <2 x float> undef, float %add, i32 0
  %e2 = insertelement <2 x float> %e1, float %add, i32 1
  %add2 = fadd fast <2 x float> %e2, <float 3.000000e+00, float -3.000000e+00>
  %e3 = insertelement <2 x float> undef, float %mul, i32 0
  %e4 = insertelement <2 x float> %e3, float %mul, i32 1
  %mul2 = fmul fast <2 x float> %add2,<float 3.000000e+00, float -3.000000e+00>
  %e5 = insertelement <2 x float> undef, float %add, i32 0
  %e6 = insertelement <2 x float> %e5, float %add, i32 1
  %add3 = fadd fast  <2 x float> %mul2, <float 3.000000e+00, float -3.000000e+00>
  %mulx = fmul fast <2 x float> %add2, %e2
  %addx = fadd fast  <2 x float> %mulx, %e4
  %e7 = insertelement <2 x float> undef, float %mul, i32 0
  %e8 = insertelement <2 x float> %e7, float %mul, i32 1
  %e9 = fmul fast <2 x float>  %addx, %add3
  store <2 x float> %e9, <2 x float>* %tmp, align 8
  %e10 = extractelement <2 x float> %add3, i32 0
  %mul3 = fmul fast float %mul, %e10
  %add4 = fadd fast float %mul3, %mul
  store float %add4, float* %arrayidx2, align 8
  %exitcond = icmp eq i64 %indvars.iv.next, 25
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
define void @foo_4s(float* %src) {
entry:
  %arrayidx1 = getelementptr inbounds float, float* %src, i64 5
  %arrayidx2 = getelementptr inbounds float, float* %src, i64 11
  %tmp = bitcast float* %arrayidx1 to <4 x float>*
  br label %for.body

; CHECK-LABEL: %for.body
; CHECK: fmla.4s {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
; CHECK: fmla.4s {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}[0]
for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx3 = getelementptr inbounds float, float* %src, i64 %indvars.iv.next
  %tmp1 = load float, float* %arrayidx3, align 8
  %add = fadd fast float %tmp1, %tmp1
  %mul = fmul fast float %add, %add
  %e1 = insertelement <4 x float> undef, float %add, i32 0
  %e2 = insertelement <4 x float> %e1, float %add, i32 1
  %add2 = fadd fast <4 x float> %e2, <float 3.000000e+00, float -3.000000e+00, float 5.000000e+00, float 7.000000e+00>
  %e3 = insertelement <4 x float> undef, float %mul, i32 0
  %e4 = insertelement <4 x float> %e3, float %mul, i32 1
  %mul2 = fmul fast <4 x float> %add2,<float 3.000000e+00, float -3.000000e+00, float 5.000000e+00, float 7.000000e+00>
  %e5 = insertelement <4 x float> undef, float %add, i32 0
  %e6 = insertelement <4 x float> %e5, float %add, i32 1
  %add3 = fadd fast  <4 x float> %mul2, <float 3.000000e+00, float -3.000000e+00, float 5.000000e+00, float 7.000000e+00> 
  %mulx = fmul fast <4 x float> %add2, %e2
  %addx = fadd fast  <4 x float> %mulx, %e4
  %e7 = insertelement <4 x float> undef, float %mul, i32 0
  %e8 = insertelement <4 x float> %e7, float %mul, i32 1
  %e9 = fmul fast <4 x float>  %addx, %add3
  store <4 x float> %e9, <4 x float>* %tmp, align 8
  %e10 = extractelement <4 x float> %add3, i32 0
  %mul3 = fmul fast float %mul, %e10
  store float %mul3, float* %arrayidx2, align 8
  %exitcond = icmp eq i64 %indvars.iv.next, 25
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
