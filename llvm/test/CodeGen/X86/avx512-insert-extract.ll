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
;CHECK: vextractpsz
;CHECK: ret
define i32 @test5(<4 x float> %x) nounwind {
  %ef = extractelement <4 x float> %x, i32 3
  %ei = bitcast float %ef to i32
  ret i32 %ei
}

;CHECK-LABEL: test6:
;CHECK: vextractpsz {{.*}}, (%rdi)
;CHECK: ret
define void @test6(<4 x float> %x, float* %out) nounwind {
  %ef = extractelement <4 x float> %x, i32 3
  store float %ef, float* %out, align 4
  ret void
}

;CHECK-LABEL: test7
;CHECK: vmovdz
;CHECK: vpermps %zmm
;CHECK: ret
define float @test7(<16 x float> %x, i32 %ind) nounwind {
  %e = extractelement <16 x float> %x, i32 %ind
  ret float %e
}

;CHECK-LABEL: test8
;CHECK: vmovqz
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
;CHECK: vmovdz
;CHECK: vpermd %zmm
;CHEKK: vmovdz  %xmm0, %eax
;CHECK: ret
define i32 @test10(<16 x i32> %x, i32 %ind) nounwind {
  %e = extractelement <16 x i32> %x, i32 %ind
  ret i32 %e
}

;CHECK-LABEL: test11
;CHECK: movl    $260
;CHECK: bextrl
;CHECK: movl    $268
;CHECK: bextrl
;CHECK: ret
define <16 x i32> @test11(<16 x i32>%a, <16 x i32>%b) {
  %cmp_res = icmp ult <16 x i32> %a, %b
  %ia = extractelement <16 x i1> %cmp_res, i32 4
  %ib = extractelement <16 x i1> %cmp_res, i32 12

  br i1 %ia, label %A, label %B

  A:
    ret <16 x i32>%b
  B:
   %c = add <16 x i32>%b, %a
  br i1 %ib, label %C, label %D
  C:
   %c1 = sub <16 x i32>%c, %a
   ret <16 x i32>%c1
  D:
   %c2 = mul <16 x i32>%c, %a
   ret <16 x i32>%c2
}
