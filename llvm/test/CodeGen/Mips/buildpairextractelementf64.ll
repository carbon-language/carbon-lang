; RUN: llc  < %s -march=mipsel | FileCheck %s -check-prefix=CHECK-EL
; RUN: llc  < %s -march=mips   | FileCheck %s -check-prefix=CHECK-EB
@a = external global i32

define double @f(i32 %a1, double %d) nounwind {
entry:
; CHECK-EL: mtc1 $6, $f12
; CHECK-EL: mtc1 $7, $f13
; CHECK-EB: mtc1 $7, $f12
; CHECK-EB: mtc1 $6, $f13
  store i32 %a1, i32* @a, align 4
  %add = fadd double %d, 2.000000e+00
  ret double %add
}

define void @f3(double %d, i32 %a1) nounwind {
entry:
; CHECK-EL: mfc1 ${{[0-9]+}}, $f12
; CHECK-EL: mfc1 $7, $f13
; CHECK-EB: mfc1 ${{[0-9]+}}, $f13
; CHECK-EB: mfc1 $7, $f12
  tail call void @f2(i32 %a1, double %d) nounwind
  ret void
}

declare void @f2(i32, double)

