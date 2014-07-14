; RUN: llc -march=mipsel -mcpu=mips32 < %s | FileCheck %s -check-prefix=ALL -check-prefix=32-NOFPXX
; RUN: llc -march=mipsel -mcpu=mips32 -mattr=fpxx < %s | FileCheck %s -check-prefix=ALL -check-prefix=32-FPXX

; RUN: llc -march=mipsel -mcpu=mips32r2 < %s | FileCheck %s -check-prefix=ALL -check-prefix=32R2-NOFPXX
; RUN: llc -march=mipsel -mcpu=mips32r2 -mattr=fpxx < %s | FileCheck %s -check-prefix=ALL -check-prefix=32R2-FPXX

; RUN: llc -march=mips64 -mcpu=mips4 < %s | FileCheck %s -check-prefix=ALL -check-prefix=4-NOFPXX
; RUN: not llc -march=mips64 -mcpu=mips4 -mattr=fpxx < %s 2>&1 | FileCheck %s -check-prefix=4-FPXX

; RUN: llc -march=mips64 -mcpu=mips64 < %s | FileCheck %s -check-prefix=ALL -check-prefix=64-NOFPXX
; RUN: not llc -march=mips64 -mcpu=mips64 -mattr=fpxx < %s 2>&1 | FileCheck %s -check-prefix=64-FPXX

; RUN-TODO: llc -march=mips64 -mcpu=mips4 -mattr=-n64,+o32 < %s | FileCheck %s -check-prefix=ALL -check-prefix=4-O32-NOFPXX
; RUN-TOOD: llc -march=mips64 -mcpu=mips4 -mattr=-n64,+o32 -mattr=fpxx < %s | FileCheck %s -check-prefix=ALL -check-prefix=4-O32-FPXX

; RUN-TODO: llc -march=mips64 -mcpu=mips64 -mattr=-n64,+o32 < %s | FileCheck %s -check-prefix=ALL -check-prefix=64-O32-NOFPXX
; RUN-TOOD: llc -march=mips64 -mcpu=mips64 -mattr=-n64,+o32 -mattr=fpxx < %s | FileCheck %s -check-prefix=ALL -check-prefix=64-O32-FPXX


; 4-FPXX:    LLVM ERROR: FPXX is not permitted for the N32/N64 ABI's.
; 64-FPXX:    LLVM ERROR: FPXX is not permitted for the N32/N64 ABI's.

define double @test1(double %d, ...) {
  ret double %d

; ALL-LABEL: test1:

; 32-NOFPXX:    mtc1    $4, $f0
; 32-NOFPXX:    mtc1    $5, $f1

; 32-FPXX:       addiu   $sp, $sp, -8
; 32-FPXX:       sw      $4, 0($sp)
; 32-FPXX:       sw      $5, 4($sp)
; 32-FPXX:       ldc1    $f0, 0($sp)

; 32R2-NOFPXX:    mtc1    $4, $f0
; 32R2-NOFPXX:    mthc1   $5, $f0

; 32R2-FPXX:    mtc1    $4, $f0
; 32R2-FPXX:    mthc1   $5, $f0

; floats/doubles are not passed in integer registers for n64, so dmtc1 is not used.
; 4-NOFPXX:    mov.d   $f0, $f12

; 64-NOFPXX:    mov.d   $f0, $f12
}

define double @test2(i32 %i, double %d) {
  ret double %d

; ALL-LABEL: test2:

; 32-NOFPXX:    mtc1    $6, $f0
; 32-NOFPXX:    mtc1    $7, $f1

; 32-FPXX:       addiu   $sp, $sp, -8
; 32-FPXX:       sw      $6, 0($sp)
; 32-FPXX:       sw      $7, 4($sp)
; 32-FPXX:       ldc1    $f0, 0($sp)

; 32R2-NOFPXX:    mtc1    $6, $f0
; 32R2-NOFPXX:    mthc1   $7, $f0

; 32R2-FPXX:    mtc1    $6, $f0
; 32R2-FPXX:    mthc1   $7, $f0

; 4-NOFPXX:    mov.d   $f0, $f13

; 64-NOFPXX:    mov.d   $f0, $f13
}

define double @test3(float %f1, float %f2, double %d) {
  ret double %d

; ALL-LABEL: test3:

; 32-NOFPXX:    mtc1    $6, $f0
; 32-NOFPXX:    mtc1    $7, $f1

; 32-FPXX:       addiu   $sp, $sp, -8
; 32-FPXX:       sw      $6, 0($sp)
; 32-FPXX:       sw      $7, 4($sp)
; 32-FPXX:       ldc1    $f0, 0($sp)

; 32R2-NOFPXX:    mtc1    $6, $f0
; 32R2-NOFPXX:    mthc1   $7, $f0

; 32R2-FPXX:    mtc1    $6, $f0
; 32R2-FPXX:    mthc1   $7, $f0

; 4-NOFPXX:    mov.d   $f0, $f14

; 64-NOFPXX:    mov.d   $f0, $f14
}

define double @test4(float %f, double %d, ...) {
  ret double %d

; ALL-LABEL: test4:

; 32-NOFPXX:    mtc1    $6, $f0
; 32-NOFPXX:    mtc1    $7, $f1

; 32-FPXX:       addiu   $sp, $sp, -8
; 32-FPXX:       sw      $6, 0($sp)
; 32-FPXX:       sw      $7, 4($sp)
; 32-FPXX:       ldc1    $f0, 0($sp)

; 32R2-NOFPXX:    mtc1    $6, $f0
; 32R2-NOFPXX:    mthc1   $7, $f0

; 32R2-FPXX:    mtc1    $6, $f0
; 32R2-FPXX:    mthc1   $7, $f0

; 4-NOFPXX:    mov.d   $f0, $f13

; 64-NOFPXX:    mov.d   $f0, $f13
}

define double @test5() {
  ret double 0.000000e+00

; ALL-LABEL: test5:

; 32-NOFPXX:    mtc1    $zero, $f0
; 32-NOFPXX:    mtc1    $zero, $f1

; 32-FPXX:    addiu   $sp, $sp, -8
; 32-FPXX:    sw      $zero, 0($sp)
; 32-FPXX:    sw      $zero, 4($sp)
; 32-FPXX:    ldc1    $f0, 0($sp)

; 32R2-NOFPXX:    mtc1    $zero, $f0
; 32R2-NOFPXX:    mthc1   $zero, $f0

; 32R2-FPXX:    mtc1    $zero, $f0
; 32R2-FPXX:    mthc1   $zero, $f0

; 4-NOFPXX:    dmtc1 $zero, $f0

; 64-NOFPXX:    dmtc1 $zero, $f0
}
