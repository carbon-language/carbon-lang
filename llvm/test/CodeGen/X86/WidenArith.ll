; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx | FileCheck %s

;CHECK-LABEL: test:
;CHECK: vaddps
;CHECK: vmulps
;CHECK: vsubps
;CHECK: vcmpltps
;CHECK: vcmpltps
;CHECK: vandps
;CHECK: vandps
;CHECK: ret
define <8 x i32> @test(<8 x float> %a, <8 x float> %b) {
 %c1 = fadd <8 x float> %a, %b
 %b1 = fmul <8 x float> %b, %a
 %d  = fsub <8 x float> %b1, %c1
 %res1 = fcmp olt <8 x float> %a, %b1
 %res2 = fcmp olt <8 x float> %c1, %d
 %andr = and <8 x i1>%res1, %res2
 %ex = zext <8 x i1> %andr to <8 x i32>
 ret <8 x i32>%ex
}


