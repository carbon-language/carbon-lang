; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s

;CHECK-LABEL: test1:
;CHECK: vinsertps
;CHECK: vinsertf32x4
;CHECK: ret
define <16 x float> @test1(<16 x float> %x, float* %br, float %y) nounwind {
  %rrr = load float* %br
  %rrr2 = insertelement <16 x float> %x, float %rrr, i32 1
  %rrr3 = insertelement <16 x float> %rrr2, float %y, i32 14
  ret <16 x float> %rrr3
}

;CHECK-LABEL: test2:
;CHECK: vinsertf32x4
;CHECK: vextractf32x4
;CHECK: vinsertf32x4
;CHECK: ret
define <8 x double> @test2(<8 x double> %x, double* %br, double %y) nounwind {
  %rrr = load double* %br
  %rrr2 = insertelement <8 x double> %x, double %rrr, i32 1
  %rrr3 = insertelement <8 x double> %rrr2, double %y, i32 6
  ret <8 x double> %rrr3
}

;CHECK-LABEL: test3:
;CHECK: vextractf32x4
;CHECK: vinsertf32x4
;CHECK: ret
define <16 x float> @test3(<16 x float> %x) nounwind {
  %eee = extractelement <16 x float> %x, i32 4
  %rrr2 = insertelement <16 x float> %x, float %eee, i32 1
  ret <16 x float> %rrr2
}

;CHECK-LABEL: test4:
;CHECK: vextracti32x4
;CHECK: vinserti32x4
;CHECK: ret
define <8 x i64> @test4(<8 x i64> %x) nounwind {
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
;CHEKK: vmovdz  %xmm0, %eax
;CHECK: ret
define i32 @test10(<16 x i32> %x, i32 %ind) nounwind {
  %e = extractelement <16 x i32> %x, i32 %ind
  ret i32 %e
}

;CHECK-LABEL: test11
;CHECK: vpcmpltud
;CKECK: kshiftlw $11
;CKECK: kshiftrw $15
;CHECK: kortestw
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
;CKECK: kshiftlw $15
;CKECK: kshiftrw $15
;CHECK: kortestw
;CHECK: ret

define i64 @test12(<16 x i64>%a, <16 x i64>%b, i64 %a1, i64 %b1) {

  %cmpvector_func.i = icmp slt <16 x i64> %a, %b
  %extract24vector_func.i = extractelement <16 x i1> %cmpvector_func.i, i32 0
  %res = select i1 %extract24vector_func.i, i64 %a1, i64 %b1
  ret i64 %res
}

;CHECK-LABEL: test13
;CHECK: cmpl
;CHECK: sbbl
;CKECK: orl $65532
;CHECK: ret
define i16 @test13(i32 %a, i32 %b) {
  %cmp_res = icmp ult i32 %a, %b
  %maskv = insertelement <16 x i1> <i1 true, i1 false, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i1 %cmp_res, i32 0
  %res = bitcast <16 x i1> %maskv to i16
  ret i16 %res
}



