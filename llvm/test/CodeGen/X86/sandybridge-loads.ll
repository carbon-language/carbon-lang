; RUN: llc -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -o - < %s | FileCheck %s

;CHECK: wideloads
;CHECK: vmovaps
;CHECK: vinsertf128
;CHECK: vmovups
;CHECK-NOT: vinsertf128
;CHECK: ret

define void @wideloads(<8 x float>* %a, <8 x float>* %b, <8 x float>* %c) nounwind uwtable noinline ssp {
  %v0 = load <8 x float>* %a, align 16  ; <---- unaligned!
  %v1 = load <8 x float>* %b, align 32  ; <---- aligned!
  %m0 = fcmp olt <8 x float> %v1, %v0
  %v2 = load <8 x float>* %c, align 16
  %m1 = fcmp olt <8 x float> %v2, %v0
  %mand = and <8 x i1> %m1, %m0
  %r = zext <8 x i1> %mand to <8 x i32>
  store <8 x i32> %r, <8 x i32>* undef, align 16
  ret void
}

