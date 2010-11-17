; RUN: llc < %s -mtriple=arm-apple-darwin | FileCheck %s
; RUN: llc < %s -march=arm -mattr=+vfp2 | FileCheck %s --check-prefix=CHECK-VFP
; RUN: llc < %s -mattr=+neon,+thumb2 -mtriple=thumbv7-apple-darwin | FileCheck %s --check-prefix=CHECK-NEON

define i32 @f1(i32 %a.s) {
;CHECK: f1:
;CHECK: moveq
entry:
    %tmp = icmp eq i32 %a.s, 4
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f2(i32 %a.s) {
;CHECK: f2:
;CHECK: movgt
entry:
    %tmp = icmp sgt i32 %a.s, 4
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f3(i32 %a.s, i32 %b.s) {
;CHECK: f3:
;CHECK: movlt
entry:
    %tmp = icmp slt i32 %a.s, %b.s
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f4(i32 %a.s, i32 %b.s) {
;CHECK: f4:
;CHECK: movle
entry:
    %tmp = icmp sle i32 %a.s, %b.s
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f5(i32 %a.u, i32 %b.u) {
;CHECK: f5:
;CHECK: movls
entry:
    %tmp = icmp ule i32 %a.u, %b.u
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f6(i32 %a.u, i32 %b.u) {
;CHECK: f6:
;CHECK: movhi
entry:
    %tmp = icmp ugt i32 %a.u, %b.u
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define double @f7(double %a, double %b) {
;CHECK: f7:
;CHECK: movlt
;CHECK: movlt
;CHECK-VFP: f7:
;CHECK-VFP: vmovmi
    %tmp = fcmp olt double %a, 1.234e+00
    %tmp1 = select i1 %tmp, double -1.000e+00, double %b
    ret double %tmp1
}

; <rdar://problem/7260094>
;
; We used to generate really horrible code for this function. The main cause was
; a lack of a custom lowering routine for an ISD::SELECT. This would result in
; two "it" blocks in the code: one for the "icmp" and another to move the index
; into the constant pool based on the value of the "icmp". If we have one "it"
; block generated, odds are good that we have close to the ideal code for this:
;
; CHECK-NEON:      _f8:
; CHECK-NEON:      movw   [[REGISTER_1:r[0-9]+]], #1123
; CHECK-NEON-NEXT: movs   [[REGISTER_2:r[0-9]+]], #0
; CHECK-NEON-NEXT: cmp    r0, [[REGISTER_1]]
; CHECK-NEON-NEXT: it     eq
; CHECK-NEON-NEXT: moveq  [[REGISTER_2]], #4
; CHECK-NEON-NEXT: adr    [[REGISTER_3:r[0-9]+]], #LCPI
; CHECK-NEON-NEXT: ldr
; CHECK-NEON:      bx

define arm_apcscc float @f8(i32 %a) nounwind {
  %tmp = icmp eq i32 %a, 1123
  %tmp1 = select i1 %tmp, float 0x3FF3BE76C0000000, float 0x40030E9A20000000
  ret float %tmp1
}
