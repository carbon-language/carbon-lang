; RUN: llc %s -o - -mattr=+avx | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

; For this test we used to optimize the <i1 true, i1 false, i1 false, i1 true>
; mask into <i32 2147483648, i32 0, i32 0, i32 2147483648> because we thought
; we would lower that into a blend where only the high bit is relevant.
; However, since the whole mask is constant, this is simplified incorrectly
; by the generic code, because it was expecting -1 in place of 2147483648.
; 
; The problem does not occur without AVX, because vselect of v4i32 is not legal
; nor custom.
;
; <rdar://problem/18675020>

; CHECK-LABEL: test:
; CHECK: vmovdqa {{.*#+}} xmm1 = [65533,124,125,14807]
; CHECK: vmovdqa {{.*#+}} xmm1 = [65535,0,0,65535]
; CHECK: ret
define void @test(<4 x i16>* %a, <4 x i16>* %b) {
body:
  %predphi = select <4 x i1> <i1 true, i1 false, i1 false, i1 true>, <4 x i16> <i16 -3, i16 545, i16 4385, i16 14807>, <4 x i16> <i16 123, i16 124, i16 125, i16 127>
  %predphi42 = select <4 x i1> <i1 true, i1 false, i1 false, i1 true>, <4 x i16> <i16 -1, i16 -1, i16 -1, i16 -1>, <4 x i16> zeroinitializer
  store <4 x i16> %predphi, <4 x i16>* %a, align 8
  store <4 x i16> %predphi42, <4 x i16>* %b, align 8
  ret void
}

; Improve code coverage.
;
; When shrinking the condition used into the select to match a blend, this
; test case exercises the path where the modified node is not the root
; of the condition.
;
; CHECK-LABEL: test2:
; CHECK:       vpslld $31, %xmm0, %xmm0
; CHECK-NEXT:  vpsrad $31, %xmm0, %xmm0
; CHECK-NEXT:  vpmovsxdq %xmm0, %xmm1
; CHECK-NEXT:  vpshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; CHECK-NEXT:  vpmovsxdq %xmm0, %xmm0
; CHECK-NEXT:  vinsertf128 $1, %xmm0, %ymm1, [[MASK:%ymm[0-9]+]]
; CHECK:       vblendvpd [[MASK]]
; CHECK:       retq
define void @test2(double** %call1559, i64 %indvars.iv4198, <4 x i1> %tmp1895) {
bb:
  %arrayidx1928 = getelementptr inbounds double*, double** %call1559, i64 %indvars.iv4198
  %tmp1888 = load double*, double** %arrayidx1928, align 8
  %predphi.v.v = select <4 x i1> %tmp1895, <4 x double> <double -5.000000e-01, double -5.000000e-01, double -5.000000e-01, double -5.000000e-01>, <4 x double> <double 5.000000e-01, double 5.000000e-01, double 5.000000e-01, double 5.000000e-01>
  %tmp1900 = bitcast double* %tmp1888 to <4 x double>*
  store <4 x double> %predphi.v.v, <4 x double>* %tmp1900, align 8
  ret void
}

; For this test, we used to optimized the conditional mask for the blend, i.e.,
; we shrunk some of its bits.
; However, this same mask was used in another select (%predphi31) that turned out
; to be optimized into a and. In that case, the conditional mask was wrong.
;
; Make sure that the and is fed by the original mask.
; 
; <rdar://problem/18819506>

; CHECK-LABEL: test3:
; Compute the mask.
;	CHECK: vpcmpeqd {{%xmm[0-9]+}}, {{%xmm[0-9]+}}, [[MASK:%xmm[0-9]+]]
; Do not shrink the bit of the mask.
; CHECK-NOT: vpslld	$31, [[MASK]], {{%xmm[0-9]+}}
; Use the mask in the blend.
; CHECK-NEXT:	vblendvps	[[MASK]], %xmm{{[0-9]+}}, %xmm{{[0-9]+}}, %xmm{{[0-9]+}}
; Use the mask in the and.
; CHECK-NEXT: vpand LCPI2_2(%rip), [[MASK]], {{%xmm[0-9]+}} 
; CHECK: retq
define void @test3(<4 x i32> %induction30, <4 x i16>* %tmp16, <4 x i16>* %tmp17,  <4 x i16> %tmp3, <4 x i16> %tmp12) {
  %tmp6 = srem <4 x i32> %induction30, <i32 3, i32 3, i32 3, i32 3>
  %tmp7 = icmp eq <4 x i32> %tmp6, zeroinitializer
  %predphi = select <4 x i1> %tmp7, <4 x i16> %tmp3, <4 x i16> %tmp12
  %predphi31 = select <4 x i1> %tmp7, <4 x i16> <i16 -1, i16 -1, i16 -1, i16 -1>, <4 x i16> zeroinitializer

  store <4 x i16> %predphi31, <4 x i16>* %tmp16, align 8
  store <4 x i16> %predphi, <4 x i16>* %tmp17, align 8
 ret void
}

; We shouldn't try to lower this directly using VSELECT because we don't have
; vpblendvb in AVX1, only in AVX2. Instead, it should be expanded.
;
; CHECK-LABEL: PR22706:
; CHECK: vpcmpgtb
; CHECK: vpcmpgtb
define <32 x i8> @PR22706(<32 x i1> %x) {
  %tmp = select <32 x i1> %x, <32 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>, <32 x i8> <i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2>
  ret <32 x i8> %tmp
}
