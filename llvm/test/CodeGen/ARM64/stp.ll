; RUN: llc < %s -march=arm64 -arm64-stp-suppress=false -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=arm64 -arm64-unscaled-mem-op=true\
; RUN:   -verify-machineinstrs | FileCheck -check-prefix=STUR_CHK %s

; CHECK: stp_int
; CHECK: stp w0, w1, [x2]
define void @stp_int(i32 %a, i32 %b, i32* nocapture %p) nounwind {
  store i32 %a, i32* %p, align 4
  %add.ptr = getelementptr inbounds i32* %p, i64 1
  store i32 %b, i32* %add.ptr, align 4
  ret void
}

; CHECK: stp_long
; CHECK: stp x0, x1, [x2]
define void @stp_long(i64 %a, i64 %b, i64* nocapture %p) nounwind {
  store i64 %a, i64* %p, align 8
  %add.ptr = getelementptr inbounds i64* %p, i64 1
  store i64 %b, i64* %add.ptr, align 8
  ret void
}

; CHECK: stp_float
; CHECK: stp s0, s1, [x0]
define void @stp_float(float %a, float %b, float* nocapture %p) nounwind {
  store float %a, float* %p, align 4
  %add.ptr = getelementptr inbounds float* %p, i64 1
  store float %b, float* %add.ptr, align 4
  ret void
}

; CHECK: stp_double
; CHECK: stp d0, d1, [x0]
define void @stp_double(double %a, double %b, double* nocapture %p) nounwind {
  store double %a, double* %p, align 8
  %add.ptr = getelementptr inbounds double* %p, i64 1
  store double %b, double* %add.ptr, align 8
  ret void
}

; Test the load/store optimizer---combine ldurs into a ldp, if appropriate
define void @stur_int(i32 %a, i32 %b, i32* nocapture %p) nounwind {
; STUR_CHK: stur_int
; STUR_CHK: stp w{{[0-9]+}}, {{w[0-9]+}}, [x{{[0-9]+}}, #-8]
; STUR_CHK-NEXT: ret
  %p1 = getelementptr inbounds i32* %p, i32 -1
  store i32 %a, i32* %p1, align 2
  %p2 = getelementptr inbounds i32* %p, i32 -2
  store i32 %b, i32* %p2, align 2
  ret void
}

define void @stur_long(i64 %a, i64 %b, i64* nocapture %p) nounwind {
; STUR_CHK: stur_long
; STUR_CHK: stp x{{[0-9]+}}, {{x[0-9]+}}, [x{{[0-9]+}}, #-16]
; STUR_CHK-NEXT: ret
  %p1 = getelementptr inbounds i64* %p, i32 -1
  store i64 %a, i64* %p1, align 2
  %p2 = getelementptr inbounds i64* %p, i32 -2
  store i64 %b, i64* %p2, align 2
  ret void
}

define void @stur_float(float %a, float %b, float* nocapture %p) nounwind {
; STUR_CHK: stur_float
; STUR_CHK: stp s{{[0-9]+}}, {{s[0-9]+}}, [x{{[0-9]+}}, #-8]
; STUR_CHK-NEXT: ret
  %p1 = getelementptr inbounds float* %p, i32 -1
  store float %a, float* %p1, align 2
  %p2 = getelementptr inbounds float* %p, i32 -2
  store float %b, float* %p2, align 2
  ret void
}

define void @stur_double(double %a, double %b, double* nocapture %p) nounwind {
; STUR_CHK: stur_double
; STUR_CHK: stp d{{[0-9]+}}, {{d[0-9]+}}, [x{{[0-9]+}}, #-16]
; STUR_CHK-NEXT: ret
  %p1 = getelementptr inbounds double* %p, i32 -1
  store double %a, double* %p1, align 2
  %p2 = getelementptr inbounds double* %p, i32 -2
  store double %b, double* %p2, align 2
  ret void
}

define void @splat_v4i32(i32 %v, i32 *%p) {
entry:

; CHECK-LABEL: splat_v4i32
; CHECK-DAG: stp w0, w0, [x1]
; CHECK-DAG: stp w0, w0, [x1, #8]
; CHECK: ret

  %p17 = insertelement <4 x i32> undef, i32 %v, i32 0
  %p18 = insertelement <4 x i32> %p17, i32 %v, i32 1
  %p19 = insertelement <4 x i32> %p18, i32 %v, i32 2
  %p20 = insertelement <4 x i32> %p19, i32 %v, i32 3
  %p21 = bitcast i32* %p to <4 x i32>*
  store <4 x i32> %p20, <4 x i32>* %p21, align 4
  ret void
}
