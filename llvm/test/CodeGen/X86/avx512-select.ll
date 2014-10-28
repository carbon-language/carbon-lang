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

; CHECK-LABEL: @select04
; CHECK: vmovaps %zmm3, %zmm1
; CHECK-NEXT: ret
; PR20677
define <16 x double> @select04(<16 x double> %a, <16 x double> %b) {
  %sel = select <16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, <16 x double> %a, <16 x double> %b
  ret <16 x double> %sel
}

; CHECK-LABEL: select05
; CHECK: kmovw   %esi, %k0
; CHECK-NEXT: kmovw   %edi, %k1
; CHECK-NEXT: korw    %k1, %k0, %k0
; CHECK-NEXT: kmovw   %k0, %eax
define i8 @select05(i8 %a.0, i8 %m) {
  %mask = bitcast i8 %m to <8 x i1>
  %a = bitcast i8 %a.0 to <8 x i1>
  %r = select <8 x i1> %mask, <8 x i1> <i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1>, <8 x i1> %a
  %res = bitcast <8 x i1> %r to i8
  ret i8 %res;
}

; CHECK-LABEL: select06
; CHECK: kmovw   %esi, %k0
; CHECK-NEXT: kmovw   %edi, %k1
; CHECK-NEXT: kandw    %k1, %k0, %k0
; CHECK-NEXT: kmovw   %k0, %eax
define i8 @select06(i8 %a.0, i8 %m) {
  %mask = bitcast i8 %m to <8 x i1>
  %a = bitcast i8 %a.0 to <8 x i1>
  %r = select <8 x i1> %mask, <8 x i1> %a, <8 x i1> zeroinitializer
  %res = bitcast <8 x i1> %r to i8
  ret i8 %res;
}

; CHECK-LABEL: select07
; CHECK-DAG:  kmovw   %edx, %k0
; CHECK-DAG:  kmovw   %edi, %k1
; CHECK-DAG:  kmovw   %esi, %k2
; CHECK: kandw %k0, %k1, %k1
; CHECK-NEXT: knotw    %k0, %k0
; CHECK-NEXT: kandw    %k0, %k2, %k0
; CHECK-NEXT: korw %k0, %k1, %k0
; CHECK-NEXT: kmovw   %k0, %eax
define i8 @select07(i8 %a.0, i8 %b.0, i8 %m) {
  %mask = bitcast i8 %m to <8 x i1>
  %a = bitcast i8 %a.0 to <8 x i1>
  %b = bitcast i8 %b.0 to <8 x i1>
  %r = select <8 x i1> %mask, <8 x i1> %a, <8 x i1> %b
  %res = bitcast <8 x i1> %r to i8
  ret i8 %res;
}
