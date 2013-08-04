; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s

;CHECK: test1
;CHECK: vinsertps
;CHECK: vinsertf32x4
;CHECK: ret
define <16 x float> @test1(<16 x float> %x, float* %br, float %y) nounwind {
  %rrr = load float* %br
  %rrr2 = insertelement <16 x float> %x, float %rrr, i32 1
  %rrr3 = insertelement <16 x float> %rrr2, float %y, i32 14
  ret <16 x float> %rrr3
}

;CHECK: test2
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

;CHECK: test3
;CHECK: vextractf32x4
;CHECK: vinsertf32x4
;CHECK: ret
define <16 x float> @test3(<16 x float> %x) nounwind {
  %eee = extractelement <16 x float> %x, i32 4
  %rrr2 = insertelement <16 x float> %x, float %eee, i32 1
  ret <16 x float> %rrr2
}

;CHECK: test4
;CHECK: vextracti32x4
;CHECK: vinserti32x4
;CHECK: ret
define <8 x i64> @test4(<8 x i64> %x) nounwind {
  %eee = extractelement <8 x i64> %x, i32 4
  %rrr2 = insertelement <8 x i64> %x, i64 %eee, i32 1
  ret <8 x i64> %rrr2
}

;CHECK: test5
;CHECK: vextractpsz
;CHECK: ret
define i32 @test5(<4 x float> %x) nounwind {
  %ef = extractelement <4 x float> %x, i32 3
  %ei = bitcast float %ef to i32
  ret i32 %ei
}

;CHECK: test6
;CHECK: vextractpsz {{.*}}, (%rdi)
;CHECK: ret
define void @test6(<4 x float> %x, float* %out) nounwind {
  %ef = extractelement <4 x float> %x, i32 3
  store float %ef, float* %out, align 4
  ret void
}

