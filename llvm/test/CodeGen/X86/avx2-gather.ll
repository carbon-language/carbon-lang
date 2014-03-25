; RUN: not llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 | FileCheck %s

declare <4 x float> @llvm.x86.avx2.gather.d.ps(<4 x float>, i8*,
                      <4 x i32>, <4 x float>, i8) nounwind readonly

define <4 x float> @test_x86_avx2_gather_d_ps(i8* %a1,
                     <4 x i32> %idx, <4 x float> %mask) {
  %res = call <4 x float> @llvm.x86.avx2.gather.d.ps(<4 x float> undef,
                            i8* %a1, <4 x i32> %idx, <4 x float> %mask, i8 2) ;
  ret <4 x float> %res
}

; CHECK: test_x86_avx2_gather_d_ps
; CHECK: vgatherdps
; CHECK-NOT: [[DST]]
; CHECK: [[DST:%xmm[0-9]+]]{{$}}
; CHECK: vmovaps
; CHECK: ret

declare <2 x double> @llvm.x86.avx2.gather.d.pd(<2 x double>, i8*,
                      <4 x i32>, <2 x double>, i8) nounwind readonly

define <2 x double> @test_x86_avx2_gather_d_pd(i8* %a1,
                     <4 x i32> %idx, <2 x double> %mask) {
  %res = call <2 x double> @llvm.x86.avx2.gather.d.pd(<2 x double> undef,
                            i8* %a1, <4 x i32> %idx, <2 x double> %mask, i8 2) ;
  ret <2 x double> %res
}

; CHECK: test_x86_avx2_gather_d_pd
; CHECK: vgatherdpd
; CHECK: vmovapd
; CHECK: ret
