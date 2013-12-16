; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl  | FileCheck %s

; CHECK-LABEL: select00
; CHECK: vmovaps
; CHECK-NEXT: LBB
define <16 x i32> @select00(i32 %a, <16 x i32> %b) nounwind {
  %cmpres = icmp eq i32 %a, 255
  %selres = select i1 %cmpres, <16 x i32> zeroinitializer, <16 x i32> %b
  %res = xor <16 x i32> %b, %selres
  ret <16 x i32> %res
}

; CHECK-LABEL: select01
; CHECK: vmovaps
; CHECK-NEXT: LBB
define <8 x i64> @select01(i32 %a, <8 x i64> %b) nounwind {
  %cmpres = icmp eq i32 %a, 255
  %selres = select i1 %cmpres, <8 x i64> zeroinitializer, <8 x i64> %b
  %res = xor <8 x i64> %b, %selres
  ret <8 x i64> %res
}

; CHECK-LABEL: @select02
; CHECK: cmpless %xmm0, %xmm3, %k1
; CHECK-NEXT: vmovss  %xmm2, {{.*}}%xmm1 {%k1}
; CHECK: ret
define float @select02(float %a, float %b, float %c, float %eps) {
  %cmp = fcmp oge float %a, %eps
  %cond = select i1 %cmp, float %c, float %b
  ret float %cond
}

; CHECK-LABEL: @select03
; CHECK: cmplesd %xmm0, %xmm3, %k1
; CHECK-NEXT: vmovsd  %xmm2, {{.*}}%xmm1 {%k1}
; CHECK: ret
define double @select03(double %a, double %b, double %c, double %eps) {
  %cmp = fcmp oge double %a, %eps
  %cond = select i1 %cmp, double %c, double %b
  ret double %cond
}
