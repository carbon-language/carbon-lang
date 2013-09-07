; RUN: llc -march=mipsel -relocation-model=pic -mno-ldc1-sdc1 -mcpu=mips32r2 \
; RUN: < %s | FileCheck %s -check-prefix=LE-PIC
; RUN: llc -march=mipsel -relocation-model=static -mno-ldc1-sdc1 < %s | \
; RUN: FileCheck %s -check-prefix=LE-STATIC
; RUN: llc -march=mips -relocation-model=pic -mno-ldc1-sdc1 < %s | \
; RUN: FileCheck %s -check-prefix=BE-PIC
; RUN: llc -march=mipsel -mcpu=mips32r2 < %s | \
; RUN: FileCheck %s -check-prefix=CHECK-LDC1-SDC1

@g0 = common global double 0.000000e+00, align 8

; LE-PIC-LABEL: test_ldc1:
; LE-PIC-DAG: lw $[[R0:[0-9]+]], 0(${{[0-9]+}})
; LE-PIC-DAG: lw $[[R1:[0-9]+]], 4(${{[0-9]+}})
; LE-PIC-DAG: mtc1 $[[R0]], $f0
; LE-PIC-DAG: mtc1 $[[R1]], $f1
; LE-STATIC-LABEL: test_ldc1:
; LE-STATIC-DAG: lui $[[R0:[0-9]+]], %hi(g0)
; LE-STATIC-DAG: lw $[[R1:[0-9]+]], %lo(g0)($[[R0]])
; LE-STATIC-DAG: addiu $[[R2:[0-9]+]], $[[R0]], %lo(g0)
; LE-STATIC-DAG: lw $[[R3:[0-9]+]], 4($[[R2]])
; LE-STATIC-DAG: mtc1 $[[R1]], $f0
; LE-STATIC-DAG: mtc1 $[[R3]], $f1
; BE-PIC-LABEL: test_ldc1:
; BE-PIC-DAG: lw $[[R0:[0-9]+]], 0(${{[0-9]+}})
; BE-PIC-DAG: lw $[[R1:[0-9]+]], 4(${{[0-9]+}})
; BE-PIC-DAG: mtc1 $[[R1]], $f0
; BE-PIC-DAG: mtc1 $[[R0]], $f1
; CHECK-LDC1-SDC1-LABEL: test_ldc1:
; CHECK-LDC1-SDC1: ldc1 $f{{[0-9]+}}

define double @test_ldc1() {
entry:
  %0 = load double* @g0, align 8
  ret double %0
}

; LE-PIC-LABEL: test_sdc1:
; LE-PIC-DAG: mfc1 $[[R0:[0-9]+]], $f12
; LE-PIC-DAG: mfc1 $[[R1:[0-9]+]], $f13
; LE-PIC-DAG: sw $[[R0]], 0(${{[0-9]+}})
; LE-PIC-DAG: sw $[[R1]], 4(${{[0-9]+}})
; LE-STATIC-LABEL: test_sdc1:
; LE-STATIC-DAG: mfc1 $[[R0:[0-9]+]], $f12
; LE-STATIC-DAG: mfc1 $[[R1:[0-9]+]], $f13
; LE-STATIC-DAG: lui $[[R2:[0-9]+]], %hi(g0)
; LE-STATIC-DAG: sw $[[R0]], %lo(g0)($[[R2]])
; LE-STATIC-DAG: addiu $[[R3:[0-9]+]], $[[R2]], %lo(g0)
; LE-STATIC-DAG: sw $[[R1]], 4($[[R3]])
; BE-PIC-LABEL: test_sdc1:
; BE-PIC-DAG: mfc1 $[[R0:[0-9]+]], $f12
; BE-PIC-DAG: mfc1 $[[R1:[0-9]+]], $f13
; BE-PIC-DAG: sw $[[R1]], 0(${{[0-9]+}})
; BE-PIC-DAG: sw $[[R0]], 4(${{[0-9]+}})
; CHECK-LDC1-SDC1-LABEL: test_sdc1:
; CHECK-LDC1-SDC1: sdc1 $f{{[0-9]+}}

define void @test_sdc1(double %a) {
entry:
  store double %a, double* @g0, align 8
  ret void
}


; LE-PIC-LABEL: test_ldxc1:
; LE-PIC-DAG: lw $[[R0:[0-9]+]], 0(${{[0-9]+}})
; LE-PIC-DAG: lw $[[R1:[0-9]+]], 4(${{[0-9]+}})
; LE-PIC-DAG: mtc1 $[[R0]], $f0
; LE-PIC-DAG: mtc1 $[[R1]], $f1
; CHECK-LDC1-SDC1-LABEL: test_ldxc1:
; CHECK-LDC1-SDC1: ldxc1 $f{{[0-9]+}}

define double @test_ldxc1(double* nocapture readonly %a, i32 %i) {
entry:
  %arrayidx = getelementptr inbounds double* %a, i32 %i
  %0 = load double* %arrayidx, align 8
  ret double %0
}

; LE-PIC-LABEL: test_sdxc1:
; LE-PIC-DAG: mfc1 $[[R0:[0-9]+]], $f12
; LE-PIC-DAG: mfc1 $[[R1:[0-9]+]], $f13
; LE-PIC-DAG: sw $[[R0]], 0(${{[0-9]+}})
; LE-PIC-DAG: sw $[[R1]], 4(${{[0-9]+}})
; CHECK-LDC1-SDC1-LABEL: test_sdxc1:
; CHECK-LDC1-SDC1: sdxc1 $f{{[0-9]+}}

define void @test_sdxc1(double %b, double* nocapture %a, i32 %i) {
entry:
  %arrayidx = getelementptr inbounds double* %a, i32 %i
  store double %b, double* %arrayidx, align 8
  ret void
}
