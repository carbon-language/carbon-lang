; RUN: llc < %s -mtriple=i386-apple-darwin10 -mcpu=corei7-avx -mattr=+avx | FileCheck %s
; RUN: llc < %s -mtriple=i386-apple-darwin10 -mcpu=corei7-avx -mattr=+avx -x86-experimental-vector-widening-legalization | FileCheck %s --check-prefix=CHECK-WIDE

define <8 x float> @foo1_8(<8 x i8> %src) {
; CHECK-LABEL: foo1_8:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpunpckhwd {{.*#+}} xmm1 = xmm0[4,4,5,5,6,6,7,7]
; CHECK-NEXT:    vpmovzxwd %xmm0, %xmm0
; CHECK-NEXT:    vpslld $24, %xmm0, %xmm0
; CHECK-NEXT:    vpsrad $24, %xmm0, %xmm0
; CHECK-NEXT:    vpslld $24, %xmm1, %xmm1
; CHECK-NEXT:    vpsrad $24, %xmm1, %xmm1
; CHECK-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; CHECK-NEXT:    vcvtdq2ps %ymm0, %ymm0
; CHECK-NEXT:    retl
;
; CHECK-WIDE-LABEL: foo1_8:
; CHECK-WIDE:       ## BB#0:
; CHECK-WIDE-NEXT:    vpmovzxbd %xmm0, %xmm1
; CHECK-WIDE-NEXT:    vpslld $24, %xmm1, %xmm1
; CHECK-WIDE-NEXT:    vpsrad $24, %xmm1, %xmm1
; CHECK-WIDE-NEXT:    vpunpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; CHECK-WIDE-NEXT:    vpunpckhwd {{.*#+}} xmm0 = xmm0[4,4,5,5,6,6,7,7]
; CHECK-WIDE-NEXT:    vpslld $24, %xmm0, %xmm0
; CHECK-WIDE-NEXT:    vpsrad $24, %xmm0, %xmm0
; CHECK-WIDE-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; CHECK-WIDE-NEXT:    vcvtdq2ps %ymm0, %ymm0
; CHECK-WIDE-NEXT:    retl
  %res = sitofp <8 x i8> %src to <8 x float>
  ret <8 x float> %res
}

define <4 x float> @foo1_4(<4 x i8> %src) {
; CHECK-LABEL: foo1_4:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpslld $24, %xmm0, %xmm0
; CHECK-NEXT:    vpsrad $24, %xmm0, %xmm0
; CHECK-NEXT:    vcvtdq2ps %xmm0, %xmm0
; CHECK-NEXT:    retl
;
; CHECK-WIDE-LABEL: foo1_4:
; CHECK-WIDE:       ## BB#0:
; CHECK-WIDE-NEXT:    vpmovzxbd %xmm0, %xmm0
; CHECK-WIDE-NEXT:    vpslld $24, %xmm0, %xmm0
; CHECK-WIDE-NEXT:    vpsrad $24, %xmm0, %xmm0
; CHECK-WIDE-NEXT:    vcvtdq2ps %xmm0, %xmm0
; CHECK-WIDE-NEXT:    retl
  %res = sitofp <4 x i8> %src to <4 x float>
  ret <4 x float> %res
}

define <8 x float> @foo2_8(<8 x i8> %src) {
; CHECK-LABEL: foo2_8:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpmovzxwd %xmm0, %xmm1
; CHECK-NEXT:    vpunpckhwd {{.*#+}} xmm0 = xmm0[4,4,5,5,6,6,7,7]
; CHECK-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; CHECK-NEXT:    vandps LCPI2_0, %ymm0, %ymm0
; CHECK-NEXT:    vcvtdq2ps %ymm0, %ymm0
; CHECK-NEXT:    retl
;
; CHECK-WIDE-LABEL: foo2_8:
; CHECK-WIDE:       ## BB#0:
; CHECK-WIDE-NEXT:    vxorps %xmm1, %xmm1, %xmm1
; CHECK-WIDE-NEXT:    vextractf128 $1, %ymm1, %xmm2
; CHECK-WIDE-NEXT:    vmovdqa {{.*#+}} xmm3 = <1,3,5,7,9,11,13,15,u,u,u,u,u,u,u,u>
; CHECK-WIDE-NEXT:    vpshufb %xmm3, %xmm2, %xmm4
; CHECK-WIDE-NEXT:    vmovdqa {{.*#+}} xmm5 = <2,6,10,14,u,u,u,u,u,u,u,u,u,u,u,u>
; CHECK-WIDE-NEXT:    vpshufb %xmm5, %xmm2, %xmm2
; CHECK-WIDE-NEXT:    vpshufb {{.*#+}} xmm6 = xmm0[4,5,6,7,u,u,u,u,u,u,u,u,u,u,u,u]
; CHECK-WIDE-NEXT:    vpunpcklbw {{.*#+}} xmm2 = xmm6[0],xmm2[0],xmm6[1],xmm2[1],xmm6[2],xmm2[2],xmm6[3],xmm2[3],xmm6[4],xmm2[4],xmm6[5],xmm2[5],xmm6[6],xmm2[6],xmm6[7],xmm2[7]
; CHECK-WIDE-NEXT:    vpunpcklbw {{.*#+}} xmm2 = xmm2[0],xmm4[0],xmm2[1],xmm4[1],xmm2[2],xmm4[2],xmm2[3],xmm4[3],xmm2[4],xmm4[4],xmm2[5],xmm4[5],xmm2[6],xmm4[6],xmm2[7],xmm4[7]
; CHECK-WIDE-NEXT:    vpshufb %xmm3, %xmm1, %xmm3
; CHECK-WIDE-NEXT:    vpshufb %xmm5, %xmm1, %xmm1
; CHECK-WIDE-NEXT:    vpmovzxbd %xmm0, %xmm0
; CHECK-WIDE-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,4,8,12,u,u,u,u,u,u,u,u,u,u,u,u]
; CHECK-WIDE-NEXT:    vpunpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; CHECK-WIDE-NEXT:    vpunpcklbw {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3],xmm0[4],xmm3[4],xmm0[5],xmm3[5],xmm0[6],xmm3[6],xmm0[7],xmm3[7]
; CHECK-WIDE-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; CHECK-WIDE-NEXT:    vcvtdq2ps %ymm0, %ymm0
; CHECK-WIDE-NEXT:    retl
  %res = uitofp <8 x i8> %src to <8 x float>
  ret <8 x float> %res
}

define <4 x float> @foo2_4(<4 x i8> %src) {
; CHECK-LABEL: foo2_4:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vandps LCPI3_0, %xmm0, %xmm0
; CHECK-NEXT:    vcvtdq2ps %xmm0, %xmm0
; CHECK-NEXT:    retl
;
; CHECK-WIDE-LABEL: foo2_4:
; CHECK-WIDE:       ## BB#0:
; CHECK-WIDE-NEXT:    vpmovzxbd %xmm0, %xmm0
; CHECK-WIDE-NEXT:    vcvtdq2ps %xmm0, %xmm0
; CHECK-WIDE-NEXT:    retl
  %res = uitofp <4 x i8> %src to <4 x float>
  ret <4 x float> %res
}

define <8 x i8> @foo3_8(<8 x float> %src) {
; CHECK-LABEL: foo3_8:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vcvttps2dq %ymm0, %ymm0
; CHECK-NEXT:    vextractf128 $1, %ymm0, %xmm1
; CHECK-NEXT:    vmovdqa {{.*#+}} xmm2 = [0,1,4,5,8,9,12,13,8,9,12,13,12,13,14,15]
; CHECK-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; CHECK-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retl
;
; CHECK-WIDE-LABEL: foo3_8:
; CHECK-WIDE:       ## BB#0:
; CHECK-WIDE-NEXT:    vpermilps {{.*#+}} xmm1 = xmm0[3,1,2,3]
; CHECK-WIDE-NEXT:    vcvttss2si %xmm1, %eax
; CHECK-WIDE-NEXT:    shll $8, %eax
; CHECK-WIDE-NEXT:    vpermilpd {{.*#+}} xmm1 = xmm0[1,0]
; CHECK-WIDE-NEXT:    vcvttss2si %xmm1, %ecx
; CHECK-WIDE-NEXT:    movzbl %cl, %ecx
; CHECK-WIDE-NEXT:    orl %eax, %ecx
; CHECK-WIDE-NEXT:    vpermilps {{.*#+}} xmm1 = xmm0[1,1,2,3]
; CHECK-WIDE-NEXT:    vcvttss2si %xmm1, %eax
; CHECK-WIDE-NEXT:    shll $8, %eax
; CHECK-WIDE-NEXT:    vcvttss2si %xmm0, %edx
; CHECK-WIDE-NEXT:    movzbl %dl, %edx
; CHECK-WIDE-NEXT:    orl %eax, %edx
; CHECK-WIDE-NEXT:    vpinsrw $0, %edx, %xmm0, %xmm1
; CHECK-WIDE-NEXT:    vpinsrw $1, %ecx, %xmm1, %xmm1
; CHECK-WIDE-NEXT:    vextractf128 $1, %ymm0, %xmm0
; CHECK-WIDE-NEXT:    vpermilps {{.*#+}} xmm2 = xmm0[1,1,2,3]
; CHECK-WIDE-NEXT:    vcvttss2si %xmm2, %eax
; CHECK-WIDE-NEXT:    shll $8, %eax
; CHECK-WIDE-NEXT:    vcvttss2si %xmm0, %ecx
; CHECK-WIDE-NEXT:    movzbl %cl, %ecx
; CHECK-WIDE-NEXT:    orl %eax, %ecx
; CHECK-WIDE-NEXT:    vpinsrw $2, %ecx, %xmm1, %xmm1
; CHECK-WIDE-NEXT:    vpermilps {{.*#+}} xmm2 = xmm0[3,1,2,3]
; CHECK-WIDE-NEXT:    vcvttss2si %xmm2, %eax
; CHECK-WIDE-NEXT:    shll $8, %eax
; CHECK-WIDE-NEXT:    vpermilpd {{.*#+}} xmm0 = xmm0[1,0]
; CHECK-WIDE-NEXT:    vcvttss2si %xmm0, %ecx
; CHECK-WIDE-NEXT:    movzbl %cl, %ecx
; CHECK-WIDE-NEXT:    orl %eax, %ecx
; CHECK-WIDE-NEXT:    vpinsrw $3, %ecx, %xmm1, %xmm0
; CHECK-WIDE-NEXT:    vzeroupper
; CHECK-WIDE-NEXT:    retl
  %res = fptosi <8 x float> %src to <8 x i8>
  ret <8 x i8> %res
}

define <4 x i8> @foo3_4(<4 x float> %src) {
; CHECK-LABEL: foo3_4:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vcvttps2dq %xmm0, %xmm0
; CHECK-NEXT:    retl
;
; CHECK-WIDE-LABEL: foo3_4:
; CHECK-WIDE:       ## BB#0:
; CHECK-WIDE-NEXT:    vpermilps {{.*#+}} xmm1 = xmm0[3,1,2,3]
; CHECK-WIDE-NEXT:    vcvttss2si %xmm1, %eax
; CHECK-WIDE-NEXT:    shll $8, %eax
; CHECK-WIDE-NEXT:    vpermilpd {{.*#+}} xmm1 = xmm0[1,0]
; CHECK-WIDE-NEXT:    vcvttss2si %xmm1, %ecx
; CHECK-WIDE-NEXT:    movzbl %cl, %ecx
; CHECK-WIDE-NEXT:    orl %eax, %ecx
; CHECK-WIDE-NEXT:    vpermilps {{.*#+}} xmm1 = xmm0[1,1,2,3]
; CHECK-WIDE-NEXT:    vcvttss2si %xmm1, %eax
; CHECK-WIDE-NEXT:    shll $8, %eax
; CHECK-WIDE-NEXT:    vcvttss2si %xmm0, %edx
; CHECK-WIDE-NEXT:    movzbl %dl, %edx
; CHECK-WIDE-NEXT:    orl %eax, %edx
; CHECK-WIDE-NEXT:    vpinsrw $0, %edx, %xmm0, %xmm0
; CHECK-WIDE-NEXT:    vpinsrw $1, %ecx, %xmm0, %xmm0
; CHECK-WIDE-NEXT:    retl
  %res = fptosi <4 x float> %src to <4 x i8>
  ret <4 x i8> %res
}

