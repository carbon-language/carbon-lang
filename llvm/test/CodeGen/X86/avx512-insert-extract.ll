; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck --check-prefix=KNL --check-prefix=CHECK %s
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=skx | FileCheck --check-prefix=SKX --check-prefix=CHECK %s

;CHECK-LABEL: test1:
;CHECK: vinsertps
;CHECK: vinsertf32x4
;CHECK: ret
define <16 x float> @test1(<16 x float> %x, float* %br, float %y) nounwind {
  %rrr = load float, float* %br
  %rrr2 = insertelement <16 x float> %x, float %rrr, i32 1
  %rrr3 = insertelement <16 x float> %rrr2, float %y, i32 14
  ret <16 x float> %rrr3
}

define <8 x double> @test2(<8 x double> %x, double* %br, double %y) nounwind {
; KNL-LABEL: test2:
; KNL:       ## BB#0:
; KNL-NEXT:    vmovhpd (%rdi), %xmm0, %xmm2
; KNL-NEXT:    vinsertf32x4 $0, %xmm2, %zmm0, %zmm0
; KNL-NEXT:    vextractf32x4 $3, %zmm0, %xmm2
; KNL-NEXT:    vmovsd %xmm1, %xmm2, %xmm1
; KNL-NEXT:    vinsertf32x4 $3, %xmm1, %zmm0, %zmm0
; KNL-NEXT:    retq
;
; SKX-LABEL: test2:
; SKX:       ## BB#0:
; SKX-NEXT:    vmovhpd (%rdi), %xmm0, %xmm2
; SKX-NEXT:    vinsertf64x2 $0, %xmm2, %zmm0, %zmm0
; SKX-NEXT:    vextractf64x2 $3, %zmm0, %xmm2
; SKX-NEXT:    vmovsd %xmm1, %xmm2, %xmm1
; SKX-NEXT:    vinsertf64x2 $3, %xmm1, %zmm0, %zmm0
; SKX-NEXT:    retq
  %rrr = load double, double* %br
  %rrr2 = insertelement <8 x double> %x, double %rrr, i32 1
  %rrr3 = insertelement <8 x double> %rrr2, double %y, i32 6
  ret <8 x double> %rrr3
}

;CHECK-LABEL: test3:
;CHECK: vextractf32x4 $1
;CHECK: vinsertf32x4 $0
;CHECK: ret
define <16 x float> @test3(<16 x float> %x) nounwind {
  %eee = extractelement <16 x float> %x, i32 4
  %rrr2 = insertelement <16 x float> %x, float %eee, i32 1
  ret <16 x float> %rrr2
}

define <8 x i64> @test4(<8 x i64> %x) nounwind {
; KNL-LABEL: test4:
; KNL:       ## BB#0:
; KNL-NEXT:    vextracti32x4 $2, %zmm0, %xmm1
; KNL-NEXT:    vmovq %xmm1, %rax
; KNL-NEXT:    vpinsrq $1, %rax, %xmm0, %xmm1
; KNL-NEXT:    vinserti32x4 $0, %xmm1, %zmm0, %zmm0
; KNL-NEXT:    retq
;
; SKX-LABEL: test4:
; SKX:       ## BB#0:
; SKX-NEXT:    vextracti64x2 $2, %zmm0, %xmm1
; SKX-NEXT:    vmovq %xmm1, %rax
; SKX-NEXT:    vpinsrq $1, %rax, %xmm0, %xmm1
; SKX-NEXT:    vinserti64x2 $0, %xmm1, %zmm0, %zmm0
; SKX-NEXT:    retq
  %eee = extractelement <8 x i64> %x, i32 4
  %rrr2 = insertelement <8 x i64> %x, i64 %eee, i32 1
  ret <8 x i64> %rrr2
}

;CHECK-LABEL: test5:
;CHECK: vextractps
;CHECK: ret
define i32 @test5(<4 x float> %x) nounwind {
  %ef = extractelement <4 x float> %x, i32 3
  %ei = bitcast float %ef to i32
  ret i32 %ei
}

;CHECK-LABEL: test6:
;CHECK: vextractps {{.*}}, (%rdi)
;CHECK: ret
define void @test6(<4 x float> %x, float* %out) nounwind {
  %ef = extractelement <4 x float> %x, i32 3
  store float %ef, float* %out, align 4
  ret void
}

;CHECK-LABEL: test7
;CHECK: vmovd
;CHECK: vpermps %zmm
;CHECK: ret
define float @test7(<16 x float> %x, i32 %ind) nounwind {
  %e = extractelement <16 x float> %x, i32 %ind
  ret float %e
}

;CHECK-LABEL: test8
;CHECK: vmovq
;CHECK: vpermpd %zmm
;CHECK: ret
define double @test8(<8 x double> %x, i32 %ind) nounwind {
  %e = extractelement <8 x double> %x, i32 %ind
  ret double %e
}

;CHECK-LABEL: test9
;CHECK: vmovd
;CHECK: vpermps %ymm
;CHECK: ret
define float @test9(<8 x float> %x, i32 %ind) nounwind {
  %e = extractelement <8 x float> %x, i32 %ind
  ret float %e
}

;CHECK-LABEL: test10
;CHECK: vmovd
;CHECK: vpermd %zmm
;CHECK: vmovd  %xmm0, %eax
;CHECK: ret
define i32 @test10(<16 x i32> %x, i32 %ind) nounwind {
  %e = extractelement <16 x i32> %x, i32 %ind
  ret i32 %e
}

;CHECK-LABEL: test11
;CHECK: vpcmpltud
;CHECK: kshiftlw $11
;CHECK: kshiftrw $15
;CHECK: testb
;CHECK: je
;CHECK: ret
;CHECK: ret
define <16 x i32> @test11(<16 x i32>%a, <16 x i32>%b) {
  %cmp_res = icmp ult <16 x i32> %a, %b
  %ia = extractelement <16 x i1> %cmp_res, i32 4
  br i1 %ia, label %A, label %B
  A:
    ret <16 x i32>%b
  B:
   %c = add <16 x i32>%b, %a
   ret <16 x i32>%c
}

;CHECK-LABEL: test12
;CHECK: vpcmpgtq
;CHECK: kshiftlw $15
;CHECK: kshiftrw $15
;CHECK: testb
;CHECK: ret

define i64 @test12(<16 x i64>%a, <16 x i64>%b, i64 %a1, i64 %b1) {

  %cmpvector_func.i = icmp slt <16 x i64> %a, %b
  %extract24vector_func.i = extractelement <16 x i1> %cmpvector_func.i, i32 0
  %res = select i1 %extract24vector_func.i, i64 %a1, i64 %b1
  ret i64 %res
}

;CHECK-LABEL: test13
;CHECK: cmpl    %esi, %edi
;CHECK: setb    %al
;CHECK: andl    $1, %eax
;CHECK: kmovw   %eax, %k0
;CHECK: movw    $-4
;CHECK: korw
define i16 @test13(i32 %a, i32 %b) {
  %cmp_res = icmp ult i32 %a, %b
  %maskv = insertelement <16 x i1> <i1 true, i1 false, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i1 %cmp_res, i32 0
  %res = bitcast <16 x i1> %maskv to i16
  ret i16 %res
}

;CHECK-LABEL: test14
;CHECK: vpcmpgtq
;KNL: kshiftlw $11
;KNL: kshiftrw $15
;KNL: testb
;SKX: kshiftlb $3
;SKX: kshiftrb $7
;SKX: testb
;CHECK: ret

define i64 @test14(<8 x i64>%a, <8 x i64>%b, i64 %a1, i64 %b1) {

  %cmpvector_func.i = icmp slt <8 x i64> %a, %b
  %extract24vector_func.i = extractelement <8 x i1> %cmpvector_func.i, i32 4
  %res = select i1 %extract24vector_func.i, i64 %a1, i64 %b1
  ret i64 %res
}

;CHECK-LABEL: test15
;CHECK: movb (%rdi), %al
;CHECK: andb $1, %al
;CHECK: movw    $-1, %ax
;CHECK: cmovew
define i16 @test15(i1 *%addr) {
  %x = load i1 , i1 * %addr, align 1
  %x1 = insertelement <16 x i1> undef, i1 %x, i32 10
  %x2 = bitcast <16 x i1>%x1 to i16
  ret i16 %x2
}

;CHECK-LABEL: test16
;CHECK: movb (%rdi), %al
;CHECK: andw $1, %ax
;CHECK: kmovw
;CHECK: kshiftlw        $10
;CHECK: korw
;CHECK: ret
define i16 @test16(i1 *%addr, i16 %a) {
  %x = load i1 , i1 * %addr, align 128
  %a1 = bitcast i16 %a to <16 x i1>
  %x1 = insertelement <16 x i1> %a1, i1 %x, i32 10
  %x2 = bitcast <16 x i1>%x1 to i16
  ret i16 %x2
}

;CHECK-LABEL: test17
;KNL: movb (%rdi), %al
;KNL: andw $1, %ax
;KNL: kshiftlw $4
;KNL: korw
;SKX: kshiftlb $4
;SKX: korb
;CHECK: ret
define i8 @test17(i1 *%addr, i8 %a) {
  %x = load i1 , i1 * %addr, align 128
  %a1 = bitcast i8 %a to <8 x i1>
  %x1 = insertelement <8 x i1> %a1, i1 %x, i32 4
  %x2 = bitcast <8 x i1>%x1 to i8
  ret i8 %x2
}

