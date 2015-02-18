; RUN: llc < %s -march=mips -mcpu=mips2 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=M2 -check-prefix=M2-M3
; RUN: llc < %s -march=mips -mcpu=mips32 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=CMOV \
; RUN:    -check-prefix=CMOV-32 -check-prefix=CMOV-32R1
; RUN: llc < %s -march=mips -mcpu=mips32r2 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=CMOV \
; RUN:    -check-prefix=CMOV-32 -check-prefix=CMOV-32R2-R5
; RUN: llc < %s -march=mips -mcpu=mips32r3 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=CMOV \
; RUN:    -check-prefix=CMOV-32 -check-prefix=CMOV-32R2-R5
; RUN: llc < %s -march=mips -mcpu=mips32r5 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=CMOV \
; RUN:    -check-prefix=CMOV-32 -check-prefix=CMOV-32R2-R5
; RUN: llc < %s -march=mips -mcpu=mips32r6 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=SEL -check-prefix=SEL-32
; RUN: llc < %s -march=mips64 -mcpu=mips3 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=M3 -check-prefix=M2-M3
; RUN: llc < %s -march=mips64 -mcpu=mips4 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=CMOV -check-prefix=CMOV-64
; RUN: llc < %s -march=mips64 -mcpu=mips64 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=CMOV -check-prefix=CMOV-64
; RUN: llc < %s -march=mips64 -mcpu=mips64r2 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=CMOV -check-prefix=CMOV-64
; RUN: llc < %s -march=mips64 -mcpu=mips64r3 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=CMOV -check-prefix=CMOV-64
; RUN: llc < %s -march=mips64 -mcpu=mips64r5 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=CMOV -check-prefix=CMOV-64
; RUN: llc < %s -march=mips64 -mcpu=mips64r6 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=SEL -check-prefix=SEL-64

define signext i1 @tst_select_i1_i1(i1 signext %s,
                                    i1 signext %x, i1 signext %y) {
entry:
  ; ALL-LABEL: tst_select_i1_i1:

  ; M2-M3:  andi    $[[T0:[0-9]+]], $4, 1
  ; M2-M3:  bnez    $[[T0]], $[[BB0:BB[0-9_]+]]
  ; M2-M3:  nop
  ; M2-M3:  move    $5, $6
  ; M2-M3:  $[[BB0]]:
  ; M2-M3:  jr      $ra
  ; M2-M3:  move    $2, $5

  ; CMOV:   andi    $[[T0:[0-9]+]], $4, 1
  ; CMOV:   movn    $6, $5, $[[T0]]
  ; CMOV:   move    $2, $6

  ; SEL:    andi    $[[T0:[0-9]+]], $4, 1
  ; SEL:    seleqz  $[[T1:[0-9]+]], $6, $[[T0]]
  ; SEL:    selnez  $[[T2:[0-9]+]], $5, $[[T0]]
  ; SEL:    or      $2, $[[T2]], $[[T1]]
  %r = select i1 %s, i1 %x, i1 %y
  ret i1 %r
}

define signext i8 @tst_select_i1_i8(i1 signext %s,
                                    i8 signext %x, i8 signext %y) {
entry:
  ; ALL-LABEL: tst_select_i1_i8:

  ; M2-M3:  andi    $[[T0:[0-9]+]], $4, 1
  ; M2-M3:  bnez    $[[T0]], $[[BB0:BB[0-9_]+]]
  ; M2-M3:  nop
  ; M2-M3:  move    $5, $6
  ; M2-M3:  $[[BB0]]:
  ; M2-M3:  jr      $ra
  ; M2-M3:  move    $2, $5

  ; CMOV:   andi    $[[T0:[0-9]+]], $4, 1
  ; CMOV:   movn    $6, $5, $[[T0]]
  ; CMOV:   move    $2, $6

  ; SEL:    andi    $[[T0:[0-9]+]], $4, 1
  ; SEL:    seleqz  $[[T1:[0-9]+]], $6, $[[T0]]
  ; SEL:    selnez  $[[T2:[0-9]+]], $5, $[[T0]]
  ; SEL:    or      $2, $[[T2]], $[[T1]]
  %r = select i1 %s, i8 %x, i8 %y
  ret i8 %r
}

define signext i32 @tst_select_i1_i32(i1 signext %s,
                                      i32 signext %x, i32 signext %y) {
entry:
  ; ALL-LABEL: tst_select_i1_i32:

  ; M2-M3:  andi    $[[T0:[0-9]+]], $4, 1
  ; M2-M3:  bnez    $[[T0]], $[[BB0:BB[0-9_]+]]
  ; M2-M3:  nop
  ; M2-M3:  move    $5, $6
  ; M2-M3:  $[[BB0]]:
  ; M2-M3:  jr      $ra
  ; M2-M3:  move    $2, $5

  ; CMOV:   andi    $[[T0:[0-9]+]], $4, 1
  ; CMOV:   movn    $6, $5, $[[T0]]
  ; CMOV:   move    $2, $6

  ; SEL:    andi    $[[T0:[0-9]+]], $4, 1
  ; SEL:    seleqz  $[[T1:[0-9]+]], $6, $[[T0]]
  ; SEL:    selnez  $[[T2:[0-9]+]], $5, $[[T0]]
  ; SEL:    or      $2, $[[T2]], $[[T1]]
  %r = select i1 %s, i32 %x, i32 %y
  ret i32 %r
}

define signext i64 @tst_select_i1_i64(i1 signext %s,
                                      i64 signext %x, i64 signext %y) {
entry:
  ; ALL-LABEL: tst_select_i1_i64:

  ; M2:     andi    $[[T0:[0-9]+]], $4, 1
  ; M2:     bnez    $[[T0]], $[[BB0:BB[0-9_]+]]
  ; M2:     nop
  ; M2:     lw      $[[T1:[0-9]+]], 16($sp)
  ; M2:     $[[BB0]]:
  ; FIXME: This branch is redundant
  ; M2:     bnez    $[[T0]], $[[BB1:BB[0-9_]+]]
  ; M2:     nop
  ; M2:     lw      $[[T2:[0-9]+]], 20($sp)
  ; M2:     $[[BB1]]:
  ; M2:     move    $2, $[[T1]]
  ; M2:     jr      $ra
  ; M2:     move    $3, $[[T2]]

  ; CMOV-32:    andi    $[[T0:[0-9]+]], $4, 1
  ; CMOV-32:    lw      $2, 16($sp)
  ; CMOV-32:    movn    $2, $6, $[[T0]]
  ; CMOV-32:    lw      $3, 20($sp)
  ; CMOV-32:    movn    $3, $7, $[[T0]]

  ; SEL-32:     andi    $[[T0:[0-9]+]], $4, 1
  ; SEL-32:     selnez  $[[T1:[0-9]+]], $6, $[[T0]]
  ; SEL-32:     lw      $[[T2:[0-9]+]], 16($sp)
  ; SEL-32:     seleqz  $[[T3:[0-9]+]], $[[T2]], $[[T0]]
  ; SEL-32:     or      $2, $[[T1]], $[[T3]]
  ; SEL-32:     selnez  $[[T4:[0-9]+]], $7, $[[T0]]
  ; SEL-32:     lw      $[[T5:[0-9]+]], 20($sp)
  ; SEL-32:     seleqz  $[[T6:[0-9]+]], $[[T5]], $[[T0]]
  ; SEL-32:     or      $3, $[[T4]], $[[T6]]

  ; M3:         andi    $[[T0:[0-9]+]], $4, 1
  ; M3:         bnez    $[[T0]], $[[BB0:BB[0-9_]+]]
  ; M3:         nop
  ; M3:         move    $5, $6
  ; M3:         $[[BB0]]:
  ; M3:         jr      $ra
  ; M3:         move    $2, $5

  ; CMOV-64:    andi    $[[T0:[0-9]+]], $4, 1
  ; CMOV-64:    movn    $6, $5, $[[T0]]
  ; CMOV-64:    move    $2, $6

  ; SEL-64:     andi    $[[T0:[0-9]+]], $4, 1
  ; FIXME: This shift is redundant
  ; SEL-64:     sll     $[[T0]], $[[T0]], 0
  ; SEL-64:     seleqz  $[[T1:[0-9]+]], $6, $[[T0]]
  ; SEL-64:     selnez  $[[T0]], $5, $[[T0]]
  ; SEL-64:     or      $2, $[[T0]], $[[T1]]
  %r = select i1 %s, i64 %x, i64 %y
  ret i64 %r
}

define float @tst_select_i1_float(i1 signext %s, float %x, float %y) {
entry:
  ; ALL-LABEL: tst_select_i1_float:

  ; M2-M3:      andi    $[[T0:[0-9]+]], $4, 1
  ; M2-M3:      bnez    $[[T0]], $[[BB0:BB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         jr      $ra
  ; M2:         mtc1    $6, $f0
  ; M3:         mov.s   $f13, $f14
  ; M2-M3:      $[[BB0]]:
  ; M2-M3:      jr      $ra
  ; M2:         mtc1    $5, $f0
  ; M3:         mov.s   $f0, $f13

  ; CMOV-32:    mtc1    $6, $f0
  ; CMOV-32:    mtc1    $5, $f1
  ; CMOV-32:    andi    $[[T0:[0-9]+]], $4, 1
  ; CMOV-32:    movn.s  $f0, $f1, $[[T0]]

  ; SEL-32:     mtc1    $5, $[[F0:f[0-9]+]]
  ; SEL-32:     mtc1    $6, $[[F1:f[0-9]+]]
  ; SEL-32:     mtc1    $4, $f0
  ; SEL-32:     sel.s   $f0, $[[F1]], $[[F0]]

  ; CMOV-64:    andi    $[[T0:[0-9]+]], $4, 1
  ; CMOV-64:    movn.s  $f14, $f13, $[[T0]]
  ; CMOV-64:    mov.s   $f0, $f14

  ; SEL-64:     mtc1    $4, $f0
  ; SEL-64:     sel.s   $f0, $f14, $f13
  %r = select i1 %s, float %x, float %y
  ret float %r
}

define float @tst_select_i1_float_reordered(float %x, float %y,
                                            i1 signext %s) {
entry:
  ; ALL-LABEL: tst_select_i1_float_reordered:

  ; M2-M3:      andi    $[[T0:[0-9]+]], $6, 1
  ; M2-M3:      bnez    $[[T0]], $[[BB0:BB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.s   $f12, $f14
  ; M3:         mov.s   $f12, $f13
  ; M2-M3:      $[[BB0]]:
  ; M2-M3:      jr      $ra
  ; M2-M3:      mov.s   $f0, $f12

  ; CMOV-32:    andi    $[[T0:[0-9]+]], $6, 1
  ; CMOV-32:    movn.s  $f14, $f12, $[[T0]]
  ; CMOV-32:    mov.s   $f0, $f14

  ; SEL-32:     mtc1    $6, $f0
  ; SEL-32:     sel.s   $f0, $f14, $f12

  ; CMOV-64:    andi    $[[T0:[0-9]+]], $6, 1
  ; CMOV-64:    movn.s  $f13, $f12, $[[T0]]
  ; CMOV-64:    mov.s   $f0, $f13

  ; SEL-64:     mtc1    $6, $f0
  ; SEL-64:     sel.s   $f0, $f13, $f12
  %r = select i1 %s, float %x, float %y
  ret float %r
}

define double @tst_select_i1_double(i1 signext %s, double %x, double %y) {
entry:
  ; ALL-LABEL: tst_select_i1_double:

  ; M2:         andi    $[[T0:[0-9]+]], $4, 1
  ; M2:         bnez    $[[T0]], $[[BB0:BB[0-9_]+]]
  ; M2:         nop
  ; M2:         ldc1    $f0, 16($sp)
  ; M2:         jr      $ra
  ; M2:         nop
  ; M2:         $[[BB0]]:
  ; M2:         mtc1    $7, $f0
  ; M2:         jr      $ra
  ; M2:         mtc1    $6, $f1

  ; CMOV-32:      mtc1    $7, $[[F0:f[0-9]+]]
  ; CMOV-32R1:    mtc1    $6, $f{{[0-9]+}}
  ; CMOV-32R2-R5: mthc1   $6, $[[F0]]
  ; CMOV-32:      andi    $[[T0:[0-9]+]], $4, 1
  ; CMOV-32:      ldc1    $f0, 16($sp)
  ; CMOV-32:      movn.d  $f0, $[[F0]], $[[T0]]

  ; SEL-32:     mtc1    $7, $[[F0:f[0-9]+]]
  ; SEL-32:     mthc1   $6, $[[F0]]
  ; SEL-32:     ldc1    $[[F1:f[0-9]+]], 16($sp)
  ; SEL-32:     mtc1    $4, $f0
  ; SEL-32:     sel.d   $f0, $[[F1]], $[[F0]]

  ; M3:         andi    $[[T0:[0-9]+]], $4, 1
  ; M3:         bnez    $[[T0]], $[[BB0:BB[0-9_]+]]
  ; M3:         nop
  ; M3:         mov.d   $f13, $f14
  ; M3:         $[[BB0]]:
  ; M3:         jr      $ra
  ; M3:         mov.d   $f0, $f13

  ; CMOV-64:    andi    $[[T0:[0-9]+]], $4, 1
  ; CMOV-64:    movn.d  $f14, $f13, $[[T0]]
  ; CMOV-64:    mov.d   $f0, $f14

  ; SEL-64:     mtc1    $4, $f0
  ; SEL-64:     sel.d   $f0, $f14, $f13
  %r = select i1 %s, double %x, double %y
  ret double %r
}

define double @tst_select_i1_double_reordered(double %x, double %y,
                                              i1 signext %s) {
entry:
  ; ALL-LABEL: tst_select_i1_double_reordered:

  ; M2:         lw      $[[T0:[0-9]+]], 16($sp)
  ; M2:         andi    $[[T1:[0-9]+]], $[[T0]], 1
  ; M2:         bnez    $[[T1]], $[[BB0:BB[0-9_]+]]
  ; M2:         nop
  ; M2:         mov.d   $f12, $f14
  ; M2:         $[[BB0]]:
  ; M2:         jr      $ra
  ; M2:         mov.d   $f0, $f12

  ; CMOV-32:    lw      $[[T0:[0-9]+]], 16($sp)
  ; CMOV-32:    andi    $[[T1:[0-9]+]], $[[T0]], 1
  ; CMOV-32:    movn.d  $f14, $f12, $[[T1]]
  ; CMOV-32:    mov.d   $f0, $f14

  ; SEL-32:     lw      $[[T0:[0-9]+]], 16($sp)
  ; SEL-32:     mtc1    $[[T0]], $f0
  ; SEL-32:     sel.d   $f0, $f14, $f12

  ; M3:         andi    $[[T0:[0-9]+]], $6, 1
  ; M3:         bnez    $[[T0]], $[[BB0:BB[0-9_]+]]
  ; M3:         nop
  ; M3:         mov.d   $f12, $f13
  ; M3:         $[[BB0]]:
  ; M3:         jr      $ra
  ; M3:         mov.d   $f0, $f12

  ; CMOV-64:    andi    $[[T0:[0-9]+]], $6, 1
  ; CMOV-64:    movn.d  $f13, $f12, $[[T0]]
  ; CMOV-64:    mov.d   $f0, $f13

  ; SEL-64:     mtc1    $6, $f0
  ; SEL-64:     sel.d   $f0, $f13, $f12
  %r = select i1 %s, double %x, double %y
  ret double %r
}

define float @tst_select_fcmp_olt_float(float %x, float %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_olt_float:

  ; M2:         c.olt.s   $f12, $f14
  ; M3:         c.olt.s   $f12, $f13
  ; M2-M3:      bc1t      $[[BB0:BB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.s     $f12, $f14
  ; M3:         mov.s     $f12, $f13
  ; M2-M3:      $[[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.s     $f0, $f12

  ; CMOV-32:    c.olt.s   $f12, $f14
  ; CMOV-32:    movt.s    $f14, $f12, $fcc0
  ; CMOV-32:    mov.s     $f0, $f14

  ; SEL-32:     cmp.lt.s  $f0, $f12, $f14
  ; SEL-32:     sel.s     $f0, $f14, $f12

  ; CMOV-64:    c.olt.s   $f12, $f13
  ; CMOV-64:    movt.s    $f13, $f12, $fcc0
  ; CMOV-64:    mov.s     $f0, $f13

  ; SEL-64:     cmp.lt.s  $f0, $f12, $f13
  ; SEL-64:     sel.s     $f0, $f13, $f12
  %s = fcmp olt float %x, %y
  %r = select i1 %s, float %x, float %y
  ret float %r
}

define float @tst_select_fcmp_ole_float(float %x, float %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_ole_float:

  ; M2:         c.ole.s   $f12, $f14
  ; M3:         c.ole.s   $f12, $f13
  ; M2-M3:      bc1t      $[[BB0:BB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.s     $f12, $f14
  ; M3:         mov.s     $f12, $f13
  ; M2-M3:      $[[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.s     $f0, $f12

  ; CMOV-32:    c.ole.s   $f12, $f14
  ; CMOV-32:    movt.s    $f14, $f12, $fcc0
  ; CMOV-32:    mov.s     $f0, $f14

  ; SEL-32:     cmp.le.s  $f0, $f12, $f14
  ; SEL-32:     sel.s     $f0, $f14, $f12

  ; CMOV-64:    c.ole.s   $f12, $f13
  ; CMOV-64:    movt.s    $f13, $f12, $fcc0
  ; CMOV-64:    mov.s     $f0, $f13

  ; SEL-64:     cmp.le.s  $f0, $f12, $f13
  ; SEL-64:     sel.s     $f0, $f13, $f12
  %s = fcmp ole float %x, %y
  %r = select i1 %s, float %x, float %y
  ret float %r
}

define float @tst_select_fcmp_ogt_float(float %x, float %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_ogt_float:

  ; M2:         c.ule.s   $f12, $f14
  ; M3:         c.ule.s   $f12, $f13
  ; M2-M3:      bc1f      $[[BB0:BB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.s     $f12, $f14
  ; M3:         mov.s     $f12, $f13
  ; M2-M3:      $[[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.s     $f0, $f12

  ; CMOV-32:    c.ule.s   $f12, $f14
  ; CMOV-32:    movf.s    $f14, $f12, $fcc0
  ; CMOV-32:    mov.s     $f0, $f14

  ; SEL-32:     cmp.lt.s  $f0, $f14, $f12
  ; SEL-32:     sel.s     $f0, $f14, $f12

  ; CMOV-64:    c.ule.s   $f12, $f13
  ; CMOV-64:    movf.s    $f13, $f12, $fcc0
  ; CMOV-64:    mov.s     $f0, $f13

  ; SEL-64:     cmp.lt.s  $f0, $f13, $f12
  ; SEL-64:     sel.s     $f0, $f13, $f12
  %s = fcmp ogt float %x, %y
  %r = select i1 %s, float %x, float %y
  ret float %r
}

define float @tst_select_fcmp_oge_float(float %x, float %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_oge_float:

  ; M2:         c.ult.s   $f12, $f14
  ; M3:         c.ult.s   $f12, $f13
  ; M2-M3:      bc1f      $[[BB0:BB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.s     $f12, $f14
  ; M3:         mov.s     $f12, $f13
  ; M2-M3:      $[[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.s     $f0, $f12

  ; CMOV-32:    c.ult.s   $f12, $f14
  ; CMOV-32:    movf.s    $f14, $f12, $fcc0
  ; CMOV-32:    mov.s     $f0, $f14

  ; SEL-32:     cmp.le.s  $f0, $f14, $f12
  ; SEL-32:     sel.s     $f0, $f14, $f12

  ; CMOV-64:    c.ult.s   $f12, $f13
  ; CMOV-64:    movf.s    $f13, $f12, $fcc0
  ; CMOV-64:    mov.s     $f0, $f13

  ; SEL-64:     cmp.le.s  $f0, $f13, $f12
  ; SEL-64:     sel.s     $f0, $f13, $f12
  %s = fcmp oge float %x, %y
  %r = select i1 %s, float %x, float %y
  ret float %r
}

define float @tst_select_fcmp_oeq_float(float %x, float %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_oeq_float:

  ; M2:         c.eq.s    $f12, $f14
  ; M3:         c.eq.s    $f12, $f13
  ; M2-M3:      bc1t      $[[BB0:BB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.s     $f12, $f14
  ; M3:         mov.s     $f12, $f13
  ; M2-M3:      $[[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.s     $f0, $f12

  ; CMOV-32:    c.eq.s    $f12, $f14
  ; CMOV-32:    movt.s    $f14, $f12, $fcc0
  ; CMOV-32:    mov.s     $f0, $f14

  ; SEL-32:     cmp.eq.s  $f0, $f12, $f14
  ; SEL-32:     sel.s     $f0, $f14, $f12

  ; CMOV-64:    c.eq.s    $f12, $f13
  ; CMOV-64:    movt.s    $f13, $f12, $fcc0
  ; CMOV-64:    mov.s     $f0, $f13

  ; SEL-64:     cmp.eq.s  $f0, $f12, $f13
  ; SEL-64:     sel.s     $f0, $f13, $f12
  %s = fcmp oeq float %x, %y
  %r = select i1 %s, float %x, float %y
  ret float %r
}

define float @tst_select_fcmp_one_float(float %x, float %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_one_float:

  ; M2:         c.ueq.s   $f12, $f14
  ; M3:         c.ueq.s   $f12, $f13
  ; M2-M3:      bc1f      $[[BB0:BB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.s     $f12, $f14
  ; M3:         mov.s     $f12, $f13
  ; M2-M3:      $[[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.s     $f0, $f12

  ; CMOV-32:    c.ueq.s   $f12, $f14
  ; CMOV-32:    movf.s    $f14, $f12, $fcc0
  ; CMOV-32:    mov.s     $f0, $f14

  ; SEL-32:     cmp.ueq.s $f0, $f12, $f14
  ; SEL-32:     mfc1      $[[T0:[0-9]+]], $f0
  ; SEL-32:     not       $[[T0]], $[[T0]]
  ; SEL-32:     mtc1      $[[T0:[0-9]+]], $f0
  ; SEL-32:     sel.s     $f0, $f14, $f12

  ; CMOV-64:    c.ueq.s   $f12, $f13
  ; CMOV-64:    movf.s    $f13, $f12, $fcc0
  ; CMOV-64:    mov.s     $f0, $f13

  ; SEL-64:     cmp.ueq.s $f0, $f12, $f13
  ; SEL-64:     mfc1      $[[T0:[0-9]+]], $f0
  ; SEL-64:     not       $[[T0]], $[[T0]]
  ; SEL-64:     mtc1      $[[T0:[0-9]+]], $f0
  ; SEL-64:     sel.s     $f0, $f13, $f12

  %s = fcmp one float %x, %y
  %r = select i1 %s, float %x, float %y
  ret float %r
}

define double @tst_select_fcmp_olt_double(double %x, double %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_olt_double:

  ; M2:         c.olt.d   $f12, $f14
  ; M3:         c.olt.d   $f12, $f13
  ; M2-M3:      bc1t      $[[BB0:BB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.d     $f12, $f14
  ; M3:         mov.d     $f12, $f13
  ; M2-M3:      $[[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.d     $f0, $f12

  ; CMOV-32:    c.olt.d   $f12, $f14
  ; CMOV-32:    movt.d    $f14, $f12, $fcc0
  ; CMOV-32:    mov.d     $f0, $f14

  ; SEL-32:     cmp.lt.d  $f0, $f12, $f14
  ; SEL-32:     sel.d     $f0, $f14, $f12

  ; CMOV-64:    c.olt.d   $f12, $f13
  ; CMOV-64:    movt.d    $f13, $f12, $fcc0
  ; CMOV-64:    mov.d     $f0, $f13

  ; SEL-64:     cmp.lt.d  $f0, $f12, $f13
  ; SEL-64:     sel.d     $f0, $f13, $f12
  %s = fcmp olt double %x, %y
  %r = select i1 %s, double %x, double %y
  ret double %r
}

define double @tst_select_fcmp_ole_double(double %x, double %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_ole_double:

  ; M2:         c.ole.d   $f12, $f14
  ; M3:         c.ole.d   $f12, $f13
  ; M2-M3:      bc1t      $[[BB0:BB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.d     $f12, $f14
  ; M3:         mov.d     $f12, $f13
  ; M2-M3:      $[[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.d     $f0, $f12

  ; CMOV-32:    c.ole.d   $f12, $f14
  ; CMOV-32:    movt.d    $f14, $f12, $fcc0
  ; CMOV-32:    mov.d     $f0, $f14

  ; SEL-32:     cmp.le.d  $f0, $f12, $f14
  ; SEL-32:     sel.d     $f0, $f14, $f12

  ; CMOV-64:    c.ole.d   $f12, $f13
  ; CMOV-64:    movt.d    $f13, $f12, $fcc0
  ; CMOV-64:    mov.d     $f0, $f13

  ; SEL-64:     cmp.le.d  $f0, $f12, $f13
  ; SEL-64:     sel.d     $f0, $f13, $f12
  %s = fcmp ole double %x, %y
  %r = select i1 %s, double %x, double %y
  ret double %r
}

define double @tst_select_fcmp_ogt_double(double %x, double %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_ogt_double:

  ; M2:         c.ule.d   $f12, $f14
  ; M3:         c.ule.d   $f12, $f13
  ; M2-M3:      bc1f      $[[BB0:BB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.d     $f12, $f14
  ; M3:         mov.d     $f12, $f13
  ; M2-M3:      $[[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.d     $f0, $f12

  ; CMOV-32:    c.ule.d   $f12, $f14
  ; CMOV-32:    movf.d    $f14, $f12, $fcc0
  ; CMOV-32:    mov.d     $f0, $f14

  ; SEL-32:     cmp.lt.d  $f0, $f14, $f12
  ; SEL-32:     sel.d     $f0, $f14, $f12

  ; CMOV-64:    c.ule.d   $f12, $f13
  ; CMOV-64:    movf.d    $f13, $f12, $fcc0
  ; CMOV-64:    mov.d     $f0, $f13

  ; SEL-64:     cmp.lt.d  $f0, $f13, $f12
  ; SEL-64:     sel.d     $f0, $f13, $f12
  %s = fcmp ogt double %x, %y
  %r = select i1 %s, double %x, double %y
  ret double %r
}

define double @tst_select_fcmp_oge_double(double %x, double %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_oge_double:

  ; M2:         c.ult.d   $f12, $f14
  ; M3:         c.ult.d   $f12, $f13
  ; M2-M3:      bc1f      $[[BB0:BB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.d     $f12, $f14
  ; M3:         mov.d     $f12, $f13
  ; M2-M3:      $[[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.d     $f0, $f12

  ; CMOV-32:    c.ult.d   $f12, $f14
  ; CMOV-32:    movf.d    $f14, $f12, $fcc0
  ; CMOV-32:    mov.d     $f0, $f14

  ; SEL-32:     cmp.le.d  $f0, $f14, $f12
  ; SEL-32:     sel.d     $f0, $f14, $f12

  ; CMOV-64:    c.ult.d   $f12, $f13
  ; CMOV-64:    movf.d    $f13, $f12, $fcc0
  ; CMOV-64:    mov.d     $f0, $f13

  ; SEL-64:     cmp.le.d  $f0, $f13, $f12
  ; SEL-64:     sel.d     $f0, $f13, $f12
  %s = fcmp oge double %x, %y
  %r = select i1 %s, double %x, double %y
  ret double %r
}

define double @tst_select_fcmp_oeq_double(double %x, double %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_oeq_double:

  ; M2:         c.eq.d    $f12, $f14
  ; M3:         c.eq.d    $f12, $f13
  ; M2-M3:      bc1t      $[[BB0:BB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.d     $f12, $f14
  ; M3:         mov.d     $f12, $f13
  ; M2-M3:      $[[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.d     $f0, $f12

  ; CMOV-32:    c.eq.d    $f12, $f14
  ; CMOV-32:    movt.d    $f14, $f12, $fcc0
  ; CMOV-32:    mov.d     $f0, $f14

  ; SEL-32:     cmp.eq.d  $f0, $f12, $f14
  ; SEL-32:     sel.d     $f0, $f14, $f12

  ; CMOV-64:    c.eq.d    $f12, $f13
  ; CMOV-64:    movt.d    $f13, $f12, $fcc0
  ; CMOV-64:    mov.d     $f0, $f13

  ; SEL-64:     cmp.eq.d  $f0, $f12, $f13
  ; SEL-64:     sel.d     $f0, $f13, $f12
  %s = fcmp oeq double %x, %y
  %r = select i1 %s, double %x, double %y
  ret double %r
}

define double @tst_select_fcmp_one_double(double %x, double %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_one_double:

  ; M2:         c.ueq.d   $f12, $f14
  ; M3:         c.ueq.d   $f12, $f13
  ; M2-M3:      bc1f      $[[BB0:BB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.d     $f12, $f14
  ; M3:         mov.d     $f12, $f13
  ; M2-M3:      $[[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.d     $f0, $f12

  ; CMOV-32:    c.ueq.d   $f12, $f14
  ; CMOV-32:    movf.d    $f14, $f12, $fcc0
  ; CMOV-32:    mov.d     $f0, $f14

  ; SEL-32:     cmp.ueq.d $f0, $f12, $f14
  ; SEL-32:     mfc1      $[[T0:[0-9]+]], $f0
  ; SEL-32:     not       $[[T0]], $[[T0]]
  ; SEL-32:     mtc1      $[[T0:[0-9]+]], $f0
  ; SEL-32:     sel.d     $f0, $f14, $f12

  ; CMOV-64:    c.ueq.d   $f12, $f13
  ; CMOV-64:    movf.d    $f13, $f12, $fcc0
  ; CMOV-64:    mov.d     $f0, $f13

  ; SEL-64:     cmp.ueq.d $f0, $f12, $f13
  ; SEL-64:     mfc1      $[[T0:[0-9]+]], $f0
  ; SEL-64:     not       $[[T0]], $[[T0]]
  ; SEL-64:     mtc1      $[[T0:[0-9]+]], $f0
  ; SEL-64:     sel.d     $f0, $f13, $f12
  %s = fcmp one double %x, %y
  %r = select i1 %s, double %x, double %y
  ret double %r
}
