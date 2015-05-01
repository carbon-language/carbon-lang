; RUN: llc < %s -mtriple=i686-unknown -mattr=+avx | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown -mattr=+avx | FileCheck %s

; Check that constant integers are correctly being truncated before float conversion

define <4 x float> @test1() {
; CHECK-LABEL: test1
; CHECK: movaps {{.*#+}} xmm0 = [-1.000000e+00,0.000000e+00,-1.000000e+00,0.000000e+00]
; CHECK-NEXT: ret
  %1 = trunc <4 x i3> <i3 -1, i3 -22, i3 7, i3 8> to <4 x i1>
  %2 = sitofp <4 x i1> %1 to <4 x float>
  ret <4 x float> %2
}
