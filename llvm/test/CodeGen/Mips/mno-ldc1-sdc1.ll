; RUN: llc -march=mipsel -relocation-model=pic -mno-ldc1-sdc1 < %s | \
; RUN: FileCheck %s -check-prefix=LE-PIC
; RUN: llc -march=mipsel -relocation-model=static -mno-ldc1-sdc1 < %s | \
; RUN: FileCheck %s -check-prefix=LE-STATIC
; RUN: llc -march=mips -relocation-model=pic -mno-ldc1-sdc1 < %s | \
; RUN: FileCheck %s -check-prefix=BE-PIC
; RUN: llc -march=mipsel < %s | FileCheck %s -check-prefix=CHECK-LDC1-SDC1

@g0 = common global double 0.000000e+00, align 8

; LE-PIC-LABEL: test_ldc1:
; LE-PIC: lwc1 $f0, 0(${{[0-9]+}})
; LE-PIC: lwc1 $f1, 4(${{[0-9]+}})
; LE-STATIC-LABEL: test_ldc1:
; LE-STATIC: lwc1 $f0, %lo(g0)(${{[0-9]+}})
; LE-STATIC: lwc1 $f1, %lo(g0+4)(${{[0-9]+}})
; BE-PIC-LABEL: test_ldc1:
; BE-PIC: lwc1 $f1, 0(${{[0-9]+}})
; BE-PIC: lwc1 $f0, 4(${{[0-9]+}})
; CHECK-LDC1-SDC1-LABEL: test_ldc1:
; CHECK-LDC1-SDC1: ldc1 $f{{[0-9]+}}

define double @test_ldc1() {
entry:
  %0 = load double* @g0, align 8
  ret double %0
}

; LE-PIC-LABEL: test_sdc1:
; LE-PIC: swc1 $f12, 0(${{[0-9]+}})
; LE-PIC: swc1 $f13, 4(${{[0-9]+}})
; LE-STATIC-LABEL: test_sdc1:
; LE-STATIC: swc1 $f12, %lo(g0)(${{[0-9]+}})
; LE-STATIC: swc1 $f13, %lo(g0+4)(${{[0-9]+}})
; BE-PIC-LABEL: test_sdc1:
; BE-PIC: swc1 $f13, 0(${{[0-9]+}})
; BE-PIC: swc1 $f12, 4(${{[0-9]+}})
; CHECK-LDC1-SDC1-LABEL: test_sdc1:
; CHECK-LDC1-SDC1: sdc1 $f{{[0-9]+}}

define void @test_sdc1(double %a) {
entry:
  store double %a, double* @g0, align 8
  ret void
}
