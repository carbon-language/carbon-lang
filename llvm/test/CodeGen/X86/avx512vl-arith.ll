; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512vl| FileCheck %s

; 256-bit

; CHECK-LABEL: vpaddq256_test
; CHECK: vpaddq %ymm{{.*}}
; CHECK: ret
define <4 x i64> @vpaddq256_test(<4 x i64> %i, <4 x i64> %j) nounwind readnone {
  %x = add <4 x i64> %i, %j
  ret <4 x i64> %x
}

; CHECK-LABEL: vpaddq256_fold_test
; CHECK: vpaddq (%rdi), %ymm{{.*}}
; CHECK: ret
define <4 x i64> @vpaddq256_fold_test(<4 x i64> %i, <4 x i64>* %j) nounwind {
  %tmp = load <4 x i64>, <4 x i64>* %j, align 4
  %x = add <4 x i64> %i, %tmp
  ret <4 x i64> %x
}

; CHECK-LABEL: vpaddq256_broadcast_test
; CHECK: vpaddq LCP{{.*}}(%rip){1to4}, %ymm{{.*}}
; CHECK: ret
define <4 x i64> @vpaddq256_broadcast_test(<4 x i64> %i) nounwind {
  %x = add <4 x i64> %i, <i64 1, i64 1, i64 1, i64 1>
  ret <4 x i64> %x
}

; CHECK-LABEL: vpaddq256_broadcast2_test
; CHECK: vpaddq (%rdi){1to4}, %ymm{{.*}}
; CHECK: ret
define <4 x i64> @vpaddq256_broadcast2_test(<4 x i64> %i, i64* %j.ptr) nounwind {
  %j = load i64, i64* %j.ptr
  %j.0 = insertelement <4 x i64> undef, i64 %j, i32 0
  %j.v = shufflevector <4 x i64> %j.0, <4 x i64> undef, <4 x i32> zeroinitializer
  %x = add <4 x i64> %i, %j.v
  ret <4 x i64> %x
}

; CHECK-LABEL: vpaddd256_test
; CHECK: vpaddd %ymm{{.*}}
; CHECK: ret
define <8 x i32> @vpaddd256_test(<8 x i32> %i, <8 x i32> %j) nounwind readnone {
  %x = add <8 x i32> %i, %j
  ret <8 x i32> %x
}

; CHECK-LABEL: vpaddd256_fold_test
; CHECK: vpaddd (%rdi), %ymm{{.*}}
; CHECK: ret
define <8 x i32> @vpaddd256_fold_test(<8 x i32> %i, <8 x i32>* %j) nounwind {
  %tmp = load <8 x i32>, <8 x i32>* %j, align 4
  %x = add <8 x i32> %i, %tmp
  ret <8 x i32> %x
}

; CHECK-LABEL: vpaddd256_broadcast_test
; CHECK: vpaddd LCP{{.*}}(%rip){1to8}, %ymm{{.*}}
; CHECK: ret
define <8 x i32> @vpaddd256_broadcast_test(<8 x i32> %i) nounwind {
  %x = add <8 x i32> %i, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <8 x i32> %x
}

; CHECK-LABEL: vpaddd256_mask_test
; CHECK: vpaddd %ymm{{.*%k[1-7].*}}
; CHECK: ret
define <8 x i32> @vpaddd256_mask_test(<8 x i32> %i, <8 x i32> %j, <8 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <8 x i32> %mask1, zeroinitializer
  %x = add <8 x i32> %i, %j
  %r = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> %i
  ret <8 x i32> %r
}

; CHECK-LABEL: vpaddd256_maskz_test
; CHECK: vpaddd %ymm{{.*{%k[1-7]} {z}.*}}
; CHECK: ret
define <8 x i32> @vpaddd256_maskz_test(<8 x i32> %i, <8 x i32> %j, <8 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <8 x i32> %mask1, zeroinitializer
  %x = add <8 x i32> %i, %j
  %r = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> zeroinitializer
  ret <8 x i32> %r
}

; CHECK-LABEL: vpaddd256_mask_fold_test
; CHECK: vpaddd (%rdi), %ymm{{.*%k[1-7]}}
; CHECK: ret
define <8 x i32> @vpaddd256_mask_fold_test(<8 x i32> %i, <8 x i32>* %j.ptr, <8 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <8 x i32> %mask1, zeroinitializer
  %j = load <8 x i32>, <8 x i32>* %j.ptr
  %x = add <8 x i32> %i, %j
  %r = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> %i
  ret <8 x i32> %r
}

; CHECK-LABEL: vpaddd256_mask_broadcast_test
; CHECK: vpaddd LCP{{.*}}(%rip){1to8}, %ymm{{.*{%k[1-7]}}}
; CHECK: ret
define <8 x i32> @vpaddd256_mask_broadcast_test(<8 x i32> %i, <8 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <8 x i32> %mask1, zeroinitializer
  %x = add <8 x i32> %i, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %r = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> %i
  ret <8 x i32> %r
}

; CHECK-LABEL: vpaddd256_maskz_fold_test
; CHECK: vpaddd (%rdi), %ymm{{.*{%k[1-7]} {z}}}
; CHECK: ret
define <8 x i32> @vpaddd256_maskz_fold_test(<8 x i32> %i, <8 x i32>* %j.ptr, <8 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <8 x i32> %mask1, zeroinitializer
  %j = load <8 x i32>, <8 x i32>* %j.ptr
  %x = add <8 x i32> %i, %j
  %r = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> zeroinitializer
  ret <8 x i32> %r
}

; CHECK-LABEL: vpaddd256_maskz_broadcast_test
; CHECK: vpaddd LCP{{.*}}(%rip){1to8}, %ymm{{.*{%k[1-7]} {z}}}
; CHECK: ret
define <8 x i32> @vpaddd256_maskz_broadcast_test(<8 x i32> %i, <8 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <8 x i32> %mask1, zeroinitializer
  %x = add <8 x i32> %i, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %r = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> zeroinitializer
  ret <8 x i32> %r
}

; CHECK-LABEL: vpsubq256_test
; CHECK: vpsubq %ymm{{.*}}
; CHECK: ret
define <4 x i64> @vpsubq256_test(<4 x i64> %i, <4 x i64> %j) nounwind readnone {
  %x = sub <4 x i64> %i, %j
  ret <4 x i64> %x
}

; CHECK-LABEL: vpsubd256_test
; CHECK: vpsubd %ymm{{.*}}
; CHECK: ret
define <8 x i32> @vpsubd256_test(<8 x i32> %i, <8 x i32> %j) nounwind readnone {
  %x = sub <8 x i32> %i, %j
  ret <8 x i32> %x
}

; CHECK-LABEL: vpmulld256_test
; CHECK: vpmulld %ymm{{.*}}
; CHECK: ret
define <8 x i32> @vpmulld256_test(<8 x i32> %i, <8 x i32> %j) {
  %x = mul <8 x i32> %i, %j
  ret <8 x i32> %x
}

; CHECK-LABEL: test_vaddpd_256
; CHECK: vaddpd{{.*}}
; CHECK: ret
define <4 x double> @test_vaddpd_256(<4 x double> %y, <4 x double> %x) {
entry:
  %add.i = fadd <4 x double> %x, %y
  ret <4 x double> %add.i
}

; CHECK-LABEL: test_fold_vaddpd_256
; CHECK: vaddpd LCP{{.*}}(%rip){{.*}}
; CHECK: ret
define <4 x double> @test_fold_vaddpd_256(<4 x double> %y) {
entry:
  %add.i = fadd <4 x double> %y, <double 4.500000e+00, double 3.400000e+00, double 4.500000e+00, double 5.600000e+00>
  ret <4 x double> %add.i
}

; CHECK-LABEL: test_broadcast_vaddpd_256
; CHECK: LCP{{.*}}(%rip){1to8}, %ymm0, %ymm0
; CHECK: ret
define <8 x float> @test_broadcast_vaddpd_256(<8 x float> %a) nounwind {
  %b = fadd <8 x float> %a, <float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000>
  ret <8 x float> %b
}

; CHECK-LABEL: test_mask_vaddps_256
; CHECK: vaddps {{%ymm[0-9]{1,2}, %ymm[0-9]{1,2}, %ymm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <8 x float> @test_mask_vaddps_256(<8 x float> %dst, <8 x float> %i,
                                        <8 x float> %j, <8 x i32> %mask1)
                                        nounwind readnone {
  %mask = icmp ne <8 x i32> %mask1, zeroinitializer
  %x = fadd <8 x float> %i, %j
  %r = select <8 x i1> %mask, <8 x float> %x, <8 x float> %dst
  ret <8 x float> %r
}

; CHECK-LABEL: test_mask_vmulps_256
; CHECK: vmulps {{%ymm[0-9]{1,2}, %ymm[0-9]{1,2}, %ymm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <8 x float> @test_mask_vmulps_256(<8 x float> %dst, <8 x float> %i,
                                        <8 x float> %j, <8 x i32> %mask1)
                                        nounwind readnone {
  %mask = icmp ne <8 x i32> %mask1, zeroinitializer
  %x = fmul <8 x float> %i, %j
  %r = select <8 x i1> %mask, <8 x float> %x, <8 x float> %dst
  ret <8 x float> %r
}

; CHECK-LABEL: test_mask_vminps_256
; CHECK: vminps {{%ymm[0-9]{1,2}, %ymm[0-9]{1,2}, %ymm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <8 x float> @test_mask_vminps_256(<8 x float> %dst, <8 x float> %i,
                                        <8 x float> %j, <8 x i32> %mask1)
                                        nounwind readnone {
  %mask = icmp ne <8 x i32> %mask1, zeroinitializer
  %cmp_res = fcmp olt <8 x float> %i, %j
  %min = select <8 x i1> %cmp_res, <8 x float> %i, <8 x float> %j
  %r = select <8 x i1> %mask, <8 x float> %min, <8 x float> %dst
  ret <8 x float> %r
}

; CHECK-LABEL: test_mask_vmaxps_256
; CHECK: vmaxps {{%ymm[0-9]{1,2}, %ymm[0-9]{1,2}, %ymm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <8 x float> @test_mask_vmaxps_256(<8 x float> %dst, <8 x float> %i,
                                        <8 x float> %j, <8 x i32> %mask1)
                                        nounwind readnone {
  %mask = icmp ne <8 x i32> %mask1, zeroinitializer
  %cmp_res = fcmp ogt <8 x float> %i, %j
  %max = select <8 x i1> %cmp_res, <8 x float> %i, <8 x float> %j
  %r = select <8 x i1> %mask, <8 x float> %max, <8 x float> %dst
  ret <8 x float> %r
}

; CHECK-LABEL: test_mask_vsubps_256
; CHECK: vsubps {{%ymm[0-9]{1,2}, %ymm[0-9]{1,2}, %ymm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <8 x float> @test_mask_vsubps_256(<8 x float> %dst, <8 x float> %i,
                                        <8 x float> %j, <8 x i32> %mask1)
                                        nounwind readnone {
  %mask = icmp ne <8 x i32> %mask1, zeroinitializer
  %x = fsub <8 x float> %i, %j
  %r = select <8 x i1> %mask, <8 x float> %x, <8 x float> %dst
  ret <8 x float> %r
}

; CHECK-LABEL: test_mask_vdivps_256
; CHECK: vdivps {{%ymm[0-9]{1,2}, %ymm[0-9]{1,2}, %ymm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <8 x float> @test_mask_vdivps_256(<8 x float> %dst, <8 x float> %i,
                                        <8 x float> %j, <8 x i32> %mask1)
                                        nounwind readnone {
  %mask = icmp ne <8 x i32> %mask1, zeroinitializer
  %x = fdiv <8 x float> %i, %j
  %r = select <8 x i1> %mask, <8 x float> %x, <8 x float> %dst
  ret <8 x float> %r
}

; CHECK-LABEL: test_mask_vmulpd_256
; CHECK: vmulpd {{%ymm[0-9]{1,2}, %ymm[0-9]{1,2}, %ymm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <4 x double> @test_mask_vmulpd_256(<4 x double> %dst, <4 x double> %i,
                                        <4 x double> %j, <4 x i64> %mask1)
                                        nounwind readnone {
  %mask = icmp ne <4 x i64> %mask1, zeroinitializer
  %x = fmul <4 x double> %i, %j
  %r = select <4 x i1> %mask, <4 x double> %x, <4 x double> %dst
  ret <4 x double> %r
}

; CHECK-LABEL: test_mask_vminpd_256
; CHECK: vminpd {{%ymm[0-9]{1,2}, %ymm[0-9]{1,2}, %ymm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <4 x double> @test_mask_vminpd_256(<4 x double> %dst, <4 x double> %i,
                                        <4 x double> %j, <4 x i64> %mask1)
                                        nounwind readnone {
  %mask = icmp ne <4 x i64> %mask1, zeroinitializer
  %cmp_res = fcmp olt <4 x double> %i, %j
  %min = select <4 x i1> %cmp_res, <4 x double> %i, <4 x double> %j
  %r = select <4 x i1> %mask, <4 x double> %min, <4 x double> %dst
  ret <4 x double> %r
}

; CHECK-LABEL: test_mask_vmaxpd_256
; CHECK: vmaxpd {{%ymm[0-9]{1,2}, %ymm[0-9]{1,2}, %ymm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <4 x double> @test_mask_vmaxpd_256(<4 x double> %dst, <4 x double> %i,
                                        <4 x double> %j, <4 x i64> %mask1)
                                        nounwind readnone {
  %mask = icmp ne <4 x i64> %mask1, zeroinitializer
  %cmp_res = fcmp ogt <4 x double> %i, %j
  %max = select <4 x i1> %cmp_res, <4 x double> %i, <4 x double> %j
  %r = select <4 x i1> %mask, <4 x double> %max, <4 x double> %dst
  ret <4 x double> %r
}

; CHECK-LABEL: test_mask_vsubpd_256
; CHECK: vsubpd {{%ymm[0-9]{1,2}, %ymm[0-9]{1,2}, %ymm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <4 x double> @test_mask_vsubpd_256(<4 x double> %dst, <4 x double> %i,
                                        <4 x double> %j, <4 x i64> %mask1)
                                        nounwind readnone {
  %mask = icmp ne <4 x i64> %mask1, zeroinitializer
  %x = fsub <4 x double> %i, %j
  %r = select <4 x i1> %mask, <4 x double> %x, <4 x double> %dst
  ret <4 x double> %r
}

; CHECK-LABEL: test_mask_vdivpd_256
; CHECK: vdivpd {{%ymm[0-9]{1,2}, %ymm[0-9]{1,2}, %ymm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <4 x double> @test_mask_vdivpd_256(<4 x double> %dst, <4 x double> %i,
                                        <4 x double> %j, <4 x i64> %mask1)
                                        nounwind readnone {
  %mask = icmp ne <4 x i64> %mask1, zeroinitializer
  %x = fdiv <4 x double> %i, %j
  %r = select <4 x i1> %mask, <4 x double> %x, <4 x double> %dst
  ret <4 x double> %r
}

; CHECK-LABEL: test_mask_vaddpd_256
; CHECK: vaddpd {{%ymm[0-9]{1,2}, %ymm[0-9]{1,2}, %ymm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <4 x double> @test_mask_vaddpd_256(<4 x double> %dst, <4 x double> %i,
                                         <4 x double> %j, <4 x i64> %mask1)
                                         nounwind readnone {
  %mask = icmp ne <4 x i64> %mask1, zeroinitializer
  %x = fadd <4 x double> %i, %j
  %r = select <4 x i1> %mask, <4 x double> %x, <4 x double> %dst
  ret <4 x double> %r
}

; CHECK-LABEL: test_maskz_vaddpd_256
; CHECK: vaddpd {{%ymm[0-9]{1,2}, %ymm[0-9]{1,2}, %ymm[0-9]{1,2} {%k[1-7]} {z}}}
; CHECK: ret
define <4 x double> @test_maskz_vaddpd_256(<4 x double> %i, <4 x double> %j,
                                          <4 x i64> %mask1) nounwind readnone {
  %mask = icmp ne <4 x i64> %mask1, zeroinitializer
  %x = fadd <4 x double> %i, %j
  %r = select <4 x i1> %mask, <4 x double> %x, <4 x double> zeroinitializer
  ret <4 x double> %r
}

; CHECK-LABEL: test_mask_fold_vaddpd_256
; CHECK: vaddpd (%rdi), {{.*%ymm[0-9]{1,2}, %ymm[0-9]{1,2} {%k[1-7]}.*}}
; CHECK: ret
define <4 x double> @test_mask_fold_vaddpd_256(<4 x double> %dst, <4 x double> %i,
                                         <4 x double>* %j,  <4 x i64> %mask1)
                                         nounwind {
  %mask = icmp ne <4 x i64> %mask1, zeroinitializer
  %tmp = load <4 x double>, <4 x double>* %j
  %x = fadd <4 x double> %i, %tmp
  %r = select <4 x i1> %mask, <4 x double> %x, <4 x double> %dst
  ret <4 x double> %r
}

; CHECK-LABEL: test_maskz_fold_vaddpd_256
; CHECK: vaddpd (%rdi), {{.*%ymm[0-9]{1,2}, %ymm[0-9]{1,2} {%k[1-7]} {z}.*}}
; CHECK: ret
define <4 x double> @test_maskz_fold_vaddpd_256(<4 x double> %i, <4 x double>* %j,
                                          <4 x i64> %mask1) nounwind {
  %mask = icmp ne <4 x i64> %mask1, zeroinitializer
  %tmp = load <4 x double>, <4 x double>* %j
  %x = fadd <4 x double> %i, %tmp
  %r = select <4 x i1> %mask, <4 x double> %x, <4 x double> zeroinitializer
  ret <4 x double> %r
}

; CHECK-LABEL: test_broadcast2_vaddpd_256
; CHECK: vaddpd (%rdi){1to4}, %ymm{{.*}}
; CHECK: ret
define <4 x double> @test_broadcast2_vaddpd_256(<4 x double> %i, double* %j) nounwind {
  %tmp = load double, double* %j
  %b = insertelement <4 x double> undef, double %tmp, i32 0
  %c = shufflevector <4 x double> %b, <4 x double> undef,
                     <4 x i32> zeroinitializer
  %x = fadd <4 x double> %c, %i
  ret <4 x double> %x
}

; CHECK-LABEL: test_mask_broadcast_vaddpd_256
; CHECK: vaddpd (%rdi){1to4}, %ymm{{.*{%k[1-7]}.*}}
; CHECK: ret
define <4 x double> @test_mask_broadcast_vaddpd_256(<4 x double> %dst, <4 x double> %i,
                                          double* %j, <4 x i64> %mask1) nounwind {
  %mask = icmp ne <4 x i64> %mask1, zeroinitializer
  %tmp = load double, double* %j
  %b = insertelement <4 x double> undef, double %tmp, i32 0
  %c = shufflevector <4 x double> %b, <4 x double> undef,
                     <4 x i32> zeroinitializer
  %x = fadd <4 x double> %c, %i
  %r = select <4 x i1> %mask, <4 x double> %x, <4 x double> %i
  ret <4 x double> %r
}

; CHECK-LABEL: test_maskz_broadcast_vaddpd_256
; CHECK: vaddpd (%rdi){1to4}, %ymm{{.*{%k[1-7]} {z}.*}}
; CHECK: ret
define <4 x double> @test_maskz_broadcast_vaddpd_256(<4 x double> %i, double* %j,
                                           <4 x i64> %mask1) nounwind {
  %mask = icmp ne <4 x i64> %mask1, zeroinitializer
  %tmp = load double, double* %j
  %b = insertelement <4 x double> undef, double %tmp, i32 0
  %c = shufflevector <4 x double> %b, <4 x double> undef,
                     <4 x i32> zeroinitializer
  %x = fadd <4 x double> %c, %i
  %r = select <4 x i1> %mask, <4 x double> %x, <4 x double> zeroinitializer
  ret <4 x double> %r
}

; 128-bit

; CHECK-LABEL: vpaddq128_test
; CHECK: vpaddq %xmm{{.*}}
; CHECK: ret
define <2 x i64> @vpaddq128_test(<2 x i64> %i, <2 x i64> %j) nounwind readnone {
  %x = add <2 x i64> %i, %j
  ret <2 x i64> %x
}

; CHECK-LABEL: vpaddq128_fold_test
; CHECK: vpaddq (%rdi), %xmm{{.*}}
; CHECK: ret
define <2 x i64> @vpaddq128_fold_test(<2 x i64> %i, <2 x i64>* %j) nounwind {
  %tmp = load <2 x i64>, <2 x i64>* %j, align 4
  %x = add <2 x i64> %i, %tmp
  ret <2 x i64> %x
}

; CHECK-LABEL: vpaddq128_broadcast2_test
; CHECK: vpaddq (%rdi){1to2}, %xmm{{.*}}
; CHECK: ret
define <2 x i64> @vpaddq128_broadcast2_test(<2 x i64> %i, i64* %j) nounwind {
  %tmp = load i64, i64* %j
  %j.0 = insertelement <2 x i64> undef, i64 %tmp, i32 0
  %j.1 = insertelement <2 x i64> %j.0, i64 %tmp, i32 1
  %x = add <2 x i64> %i, %j.1
  ret <2 x i64> %x
}

; CHECK-LABEL: vpaddd128_test
; CHECK: vpaddd %xmm{{.*}}
; CHECK: ret
define <4 x i32> @vpaddd128_test(<4 x i32> %i, <4 x i32> %j) nounwind readnone {
  %x = add <4 x i32> %i, %j
  ret <4 x i32> %x
}

; CHECK-LABEL: vpaddd128_fold_test
; CHECK: vpaddd (%rdi), %xmm{{.*}}
; CHECK: ret
define <4 x i32> @vpaddd128_fold_test(<4 x i32> %i, <4 x i32>* %j) nounwind {
  %tmp = load <4 x i32>, <4 x i32>* %j, align 4
  %x = add <4 x i32> %i, %tmp
  ret <4 x i32> %x
}

; CHECK-LABEL: vpaddd128_broadcast_test
; CHECK: vpaddd LCP{{.*}}(%rip){1to4}, %xmm{{.*}}
; CHECK: ret
define <4 x i32> @vpaddd128_broadcast_test(<4 x i32> %i) nounwind {
  %x = add <4 x i32> %i, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %x
}

; CHECK-LABEL: vpaddd128_mask_test
; CHECK: vpaddd %xmm{{.*%k[1-7].*}}
; CHECK: ret
define <4 x i32> @vpaddd128_mask_test(<4 x i32> %i, <4 x i32> %j, <4 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <4 x i32> %mask1, zeroinitializer
  %x = add <4 x i32> %i, %j
  %r = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> %i
  ret <4 x i32> %r
}

; CHECK-LABEL: vpaddd128_maskz_test
; CHECK: vpaddd %xmm{{.*{%k[1-7]} {z}.*}}
; CHECK: ret
define <4 x i32> @vpaddd128_maskz_test(<4 x i32> %i, <4 x i32> %j, <4 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <4 x i32> %mask1, zeroinitializer
  %x = add <4 x i32> %i, %j
  %r = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> zeroinitializer
  ret <4 x i32> %r
}

; CHECK-LABEL: vpaddd128_mask_fold_test
; CHECK: vpaddd (%rdi), %xmm{{.*%k[1-7]}}
; CHECK: ret
define <4 x i32> @vpaddd128_mask_fold_test(<4 x i32> %i, <4 x i32>* %j.ptr, <4 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <4 x i32> %mask1, zeroinitializer
  %j = load <4 x i32>, <4 x i32>* %j.ptr
  %x = add <4 x i32> %i, %j
  %r = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> %i
  ret <4 x i32> %r
}

; CHECK-LABEL: vpaddd128_mask_broadcast_test
; CHECK: vpaddd LCP{{.*}}(%rip){1to4}, %xmm{{.*{%k[1-7]}}}
; CHECK: ret
define <4 x i32> @vpaddd128_mask_broadcast_test(<4 x i32> %i, <4 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <4 x i32> %mask1, zeroinitializer
  %x = add <4 x i32> %i, <i32 1, i32 1, i32 1, i32 1>
  %r = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> %i
  ret <4 x i32> %r
}

; CHECK-LABEL: vpaddd128_maskz_fold_test
; CHECK: vpaddd (%rdi), %xmm{{.*{%k[1-7]} {z}}}
; CHECK: ret
define <4 x i32> @vpaddd128_maskz_fold_test(<4 x i32> %i, <4 x i32>* %j.ptr, <4 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <4 x i32> %mask1, zeroinitializer
  %j = load <4 x i32>, <4 x i32>* %j.ptr
  %x = add <4 x i32> %i, %j
  %r = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> zeroinitializer
  ret <4 x i32> %r
}

; CHECK-LABEL: vpaddd128_maskz_broadcast_test
; CHECK: vpaddd LCP{{.*}}(%rip){1to4}, %xmm{{.*{%k[1-7]} {z}}}
; CHECK: ret
define <4 x i32> @vpaddd128_maskz_broadcast_test(<4 x i32> %i, <4 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <4 x i32> %mask1, zeroinitializer
  %x = add <4 x i32> %i, <i32 1, i32 1, i32 1, i32 1>
  %r = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> zeroinitializer
  ret <4 x i32> %r
}

; CHECK-LABEL: vpsubq128_test
; CHECK: vpsubq %xmm{{.*}}
; CHECK: ret
define <2 x i64> @vpsubq128_test(<2 x i64> %i, <2 x i64> %j) nounwind readnone {
  %x = sub <2 x i64> %i, %j
  ret <2 x i64> %x
}

; CHECK-LABEL: vpsubd128_test
; CHECK: vpsubd %xmm{{.*}}
; CHECK: ret
define <4 x i32> @vpsubd128_test(<4 x i32> %i, <4 x i32> %j) nounwind readnone {
  %x = sub <4 x i32> %i, %j
  ret <4 x i32> %x
}

; CHECK-LABEL: vpmulld128_test
; CHECK: vpmulld %xmm{{.*}}
; CHECK: ret
define <4 x i32> @vpmulld128_test(<4 x i32> %i, <4 x i32> %j) {
  %x = mul <4 x i32> %i, %j
  ret <4 x i32> %x
}

; CHECK-LABEL: test_vaddpd_128
; CHECK: vaddpd{{.*}}
; CHECK: ret
define <2 x double> @test_vaddpd_128(<2 x double> %y, <2 x double> %x) {
entry:
  %add.i = fadd <2 x double> %x, %y
  ret <2 x double> %add.i
}

; CHECK-LABEL: test_fold_vaddpd_128
; CHECK: vaddpd LCP{{.*}}(%rip){{.*}}
; CHECK: ret
define <2 x double> @test_fold_vaddpd_128(<2 x double> %y) {
entry:
  %add.i = fadd <2 x double> %y, <double 4.500000e+00, double 3.400000e+00>
  ret <2 x double> %add.i
}

; CHECK-LABEL: test_broadcast_vaddpd_128
; CHECK: LCP{{.*}}(%rip){1to4}, %xmm0, %xmm0
; CHECK: ret
define <4 x float> @test_broadcast_vaddpd_128(<4 x float> %a) nounwind {
  %b = fadd <4 x float> %a, <float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000>
  ret <4 x float> %b
}

; CHECK-LABEL: test_mask_vaddps_128
; CHECK: vaddps {{%xmm[0-9]{1,2}, %xmm[0-9]{1,2}, %xmm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <4 x float> @test_mask_vaddps_128(<4 x float> %dst, <4 x float> %i,
                                        <4 x float> %j, <4 x i32> %mask1)
                                        nounwind readnone {
  %mask = icmp ne <4 x i32> %mask1, zeroinitializer
  %x = fadd <4 x float> %i, %j
  %r = select <4 x i1> %mask, <4 x float> %x, <4 x float> %dst
  ret <4 x float> %r
}

; CHECK-LABEL: test_mask_vmulps_128
; CHECK: vmulps {{%xmm[0-9]{1,2}, %xmm[0-9]{1,2}, %xmm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <4 x float> @test_mask_vmulps_128(<4 x float> %dst, <4 x float> %i,
                                        <4 x float> %j, <4 x i32> %mask1)
                                        nounwind readnone {
  %mask = icmp ne <4 x i32> %mask1, zeroinitializer
  %x = fmul <4 x float> %i, %j
  %r = select <4 x i1> %mask, <4 x float> %x, <4 x float> %dst
  ret <4 x float> %r
}

; CHECK-LABEL: test_mask_vminps_128
; CHECK: vminps {{%xmm[0-9]{1,2}, %xmm[0-9]{1,2}, %xmm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <4 x float> @test_mask_vminps_128(<4 x float> %dst, <4 x float> %i,
                                        <4 x float> %j, <4 x i32> %mask1)
                                        nounwind readnone {
  %mask = icmp ne <4 x i32> %mask1, zeroinitializer
  %cmp_res = fcmp olt <4 x float> %i, %j
  %min = select <4 x i1> %cmp_res, <4 x float> %i, <4 x float> %j
  %r = select <4 x i1> %mask, <4 x float> %min, <4 x float> %dst
  ret <4 x float> %r
}

; CHECK-LABEL: test_mask_vmaxps_128
; CHECK: vmaxps {{%xmm[0-9]{1,2}, %xmm[0-9]{1,2}, %xmm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <4 x float> @test_mask_vmaxps_128(<4 x float> %dst, <4 x float> %i,
                                        <4 x float> %j, <4 x i32> %mask1)
                                        nounwind readnone {
  %mask = icmp ne <4 x i32> %mask1, zeroinitializer
  %cmp_res = fcmp ogt <4 x float> %i, %j
  %max = select <4 x i1> %cmp_res, <4 x float> %i, <4 x float> %j
  %r = select <4 x i1> %mask, <4 x float> %max, <4 x float> %dst
  ret <4 x float> %r
}

; CHECK-LABEL: test_mask_vsubps_128
; CHECK: vsubps {{%xmm[0-9]{1,2}, %xmm[0-9]{1,2}, %xmm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <4 x float> @test_mask_vsubps_128(<4 x float> %dst, <4 x float> %i,
                                        <4 x float> %j, <4 x i32> %mask1)
                                        nounwind readnone {
  %mask = icmp ne <4 x i32> %mask1, zeroinitializer
  %x = fsub <4 x float> %i, %j
  %r = select <4 x i1> %mask, <4 x float> %x, <4 x float> %dst
  ret <4 x float> %r
}


; CHECK-LABEL: test_mask_vdivps_128
; CHECK: vdivps {{%xmm[0-9]{1,2}, %xmm[0-9]{1,2}, %xmm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <4 x float> @test_mask_vdivps_128(<4 x float> %dst, <4 x float> %i,
                                        <4 x float> %j, <4 x i32> %mask1)
                                        nounwind readnone {
  %mask = icmp ne <4 x i32> %mask1, zeroinitializer
  %x = fdiv <4 x float> %i, %j
  %r = select <4 x i1> %mask, <4 x float> %x, <4 x float> %dst
  ret <4 x float> %r
}

; CHECK-LABEL: test_mask_vmulpd_128
; CHECK: vmulpd {{%xmm[0-9]{1,2}, %xmm[0-9]{1,2}, %xmm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <2 x double> @test_mask_vmulpd_128(<2 x double> %dst, <2 x double> %i,
                                        <2 x double> %j, <2 x i64> %mask1)
                                        nounwind readnone {
  %mask = icmp ne <2 x i64> %mask1, zeroinitializer
  %x = fmul <2 x double> %i, %j
  %r = select <2 x i1> %mask, <2 x double> %x, <2 x double> %dst
  ret <2 x double> %r
}

; CHECK-LABEL: test_mask_vminpd_128
; CHECK: vminpd {{%xmm[0-9]{1,2}, %xmm[0-9]{1,2}, %xmm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <2 x double> @test_mask_vminpd_128(<2 x double> %dst, <2 x double> %i,
                                        <2 x double> %j, <2 x i64> %mask1)
                                        nounwind readnone {
  %mask = icmp ne <2 x i64> %mask1, zeroinitializer
  %cmp_res = fcmp olt <2 x double> %i, %j
  %min = select <2 x i1> %cmp_res, <2 x double> %i, <2 x double> %j
  %r = select <2 x i1> %mask, <2 x double> %min, <2 x double> %dst
  ret <2 x double> %r
}

; CHECK-LABEL: test_mask_vmaxpd_128
; CHECK: vmaxpd {{%xmm[0-9]{1,2}, %xmm[0-9]{1,2}, %xmm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <2 x double> @test_mask_vmaxpd_128(<2 x double> %dst, <2 x double> %i,
                                        <2 x double> %j, <2 x i64> %mask1)
                                        nounwind readnone {
  %mask = icmp ne <2 x i64> %mask1, zeroinitializer
  %cmp_res = fcmp ogt <2 x double> %i, %j
  %max = select <2 x i1> %cmp_res, <2 x double> %i, <2 x double> %j
  %r = select <2 x i1> %mask, <2 x double> %max, <2 x double> %dst
  ret <2 x double> %r
}

; CHECK-LABEL: test_mask_vsubpd_128
; CHECK: vsubpd {{%xmm[0-9]{1,2}, %xmm[0-9]{1,2}, %xmm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <2 x double> @test_mask_vsubpd_128(<2 x double> %dst, <2 x double> %i,
                                        <2 x double> %j, <2 x i64> %mask1)
                                        nounwind readnone {
  %mask = icmp ne <2 x i64> %mask1, zeroinitializer
  %x = fsub <2 x double> %i, %j
  %r = select <2 x i1> %mask, <2 x double> %x, <2 x double> %dst
  ret <2 x double> %r
}

; CHECK-LABEL: test_mask_vdivpd_128
; CHECK: vdivpd {{%xmm[0-9]{1,2}, %xmm[0-9]{1,2}, %xmm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <2 x double> @test_mask_vdivpd_128(<2 x double> %dst, <2 x double> %i,
                                        <2 x double> %j, <2 x i64> %mask1)
                                        nounwind readnone {
  %mask = icmp ne <2 x i64> %mask1, zeroinitializer
  %x = fdiv <2 x double> %i, %j
  %r = select <2 x i1> %mask, <2 x double> %x, <2 x double> %dst
  ret <2 x double> %r
}

; CHECK-LABEL: test_mask_vaddpd_128
; CHECK: vaddpd {{%xmm[0-9]{1,2}, %xmm[0-9]{1,2}, %xmm[0-9]{1,2} {%k[1-7]}}}
; CHECK: ret
define <2 x double> @test_mask_vaddpd_128(<2 x double> %dst, <2 x double> %i,
                                         <2 x double> %j, <2 x i64> %mask1)
                                         nounwind readnone {
  %mask = icmp ne <2 x i64> %mask1, zeroinitializer
  %x = fadd <2 x double> %i, %j
  %r = select <2 x i1> %mask, <2 x double> %x, <2 x double> %dst
  ret <2 x double> %r
}

; CHECK-LABEL: test_maskz_vaddpd_128
; CHECK: vaddpd {{%xmm[0-9]{1,2}, %xmm[0-9]{1,2}, %xmm[0-9]{1,2} {%k[1-7]} {z}}}
; CHECK: ret
define <2 x double> @test_maskz_vaddpd_128(<2 x double> %i, <2 x double> %j,
                                          <2 x i64> %mask1) nounwind readnone {
  %mask = icmp ne <2 x i64> %mask1, zeroinitializer
  %x = fadd <2 x double> %i, %j
  %r = select <2 x i1> %mask, <2 x double> %x, <2 x double> zeroinitializer
  ret <2 x double> %r
}

; CHECK-LABEL: test_mask_fold_vaddpd_128
; CHECK: vaddpd (%rdi), {{.*%xmm[0-9]{1,2}, %xmm[0-9]{1,2} {%k[1-7]}.*}}
; CHECK: ret
define <2 x double> @test_mask_fold_vaddpd_128(<2 x double> %dst, <2 x double> %i,
                                         <2 x double>* %j,  <2 x i64> %mask1)
                                         nounwind {
  %mask = icmp ne <2 x i64> %mask1, zeroinitializer
  %tmp = load <2 x double>, <2 x double>* %j
  %x = fadd <2 x double> %i, %tmp
  %r = select <2 x i1> %mask, <2 x double> %x, <2 x double> %dst
  ret <2 x double> %r
}

; CHECK-LABEL: test_maskz_fold_vaddpd_128
; CHECK: vaddpd (%rdi), {{.*%xmm[0-9]{1,2}, %xmm[0-9]{1,2} {%k[1-7]} {z}.*}}
; CHECK: ret
define <2 x double> @test_maskz_fold_vaddpd_128(<2 x double> %i, <2 x double>* %j,
                                          <2 x i64> %mask1) nounwind {
  %mask = icmp ne <2 x i64> %mask1, zeroinitializer
  %tmp = load <2 x double>, <2 x double>* %j
  %x = fadd <2 x double> %i, %tmp
  %r = select <2 x i1> %mask, <2 x double> %x, <2 x double> zeroinitializer
  ret <2 x double> %r
}

; CHECK-LABEL: test_broadcast2_vaddpd_128
; CHECK: vaddpd (%rdi){1to2}, %xmm{{.*}}
; CHECK: ret
define <2 x double> @test_broadcast2_vaddpd_128(<2 x double> %i, double* %j) nounwind {
  %tmp = load double, double* %j
  %j.0 = insertelement <2 x double> undef, double %tmp, i64 0
  %j.1 = insertelement <2 x double> %j.0, double %tmp, i64 1
  %x = fadd <2 x double> %j.1, %i
  ret <2 x double> %x
}

; CHECK-LABEL: test_mask_broadcast_vaddpd_128
; CHECK: vaddpd (%rdi){1to2}, %xmm{{.*{%k[1-7]}.*}}
; CHECK: ret
define <2 x double> @test_mask_broadcast_vaddpd_128(<2 x double> %dst, <2 x double> %i,
                                          double* %j, <2 x i64> %mask1)
                                          nounwind {
  %mask = icmp ne <2 x i64> %mask1, zeroinitializer
  %tmp = load double, double* %j
  %j.0 = insertelement <2 x double> undef, double %tmp, i64 0
  %j.1 = insertelement <2 x double> %j.0, double %tmp, i64 1
  %x = fadd <2 x double> %j.1, %i
  %r = select <2 x i1> %mask, <2 x double> %x, <2 x double> %i
  ret <2 x double> %r
}

; CHECK-LABEL: test_maskz_broadcast_vaddpd_128
; CHECK: vaddpd (%rdi){1to2}, %xmm{{.*{%k[1-7]} {z}.*}}
; CHECK: ret
define <2 x double> @test_maskz_broadcast_vaddpd_128(<2 x double> %i, double* %j,
                                           <2 x i64> %mask1) nounwind {
  %mask = icmp ne <2 x i64> %mask1, zeroinitializer
  %tmp = load double, double* %j
  %j.0 = insertelement <2 x double> undef, double %tmp, i64 0
  %j.1 = insertelement <2 x double> %j.0, double %tmp, i64 1
  %x = fadd <2 x double> %j.1, %i
  %r = select <2 x i1> %mask, <2 x double> %x, <2 x double> zeroinitializer
  ret <2 x double> %r
}
