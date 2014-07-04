; Test all important variants of the 'ret' instruction.
;
; For non-void returns it is necessary to have something to return so we also
; test constant generation here.
;
; We'll test pointer returns in a separate file since the relocation model
; affects it.

; RUN: llc -march=mips   -mcpu=mips32   < %s | FileCheck %s -check-prefix=ALL -check-prefix=GPR32 -check-prefix=NO-MTHC1
; RUN: llc -march=mips   -mcpu=mips32r2 < %s | FileCheck %s -check-prefix=ALL -check-prefix=GPR32 -check-prefix=MTHC1
; RUN: llc -march=mips64 -mcpu=mips4    < %s | FileCheck %s -check-prefix=ALL -check-prefix=GPR64 -check-prefix=DMTC1
; RUN: llc -march=mips64 -mcpu=mips64   < %s | FileCheck %s -check-prefix=ALL -check-prefix=GPR64 -check-prefix=DMTC1
; RUN: llc -march=mips64 -mcpu=mips64r2 < %s | FileCheck %s -check-prefix=ALL -check-prefix=GPR64 -check-prefix=DMTC1

define void @ret_void() {
; ALL-LABEL: ret_void:
; ALL:           jr $ra
  ret void
}

define i8 @ret_i8() {
; ALL-LABEL: ret_i8:
; ALL-DAG:       jr $ra
; ALL-DAG:       addiu $2, $zero, 3
  ret i8 3
}

define i16 @ret_i16_3() {
; ALL-LABEL: ret_i16_3:
; ALL-DAG:       jr $ra
; ALL-DAG:       addiu $2, $zero, 3
  ret i16 3
}

define i16 @ret_i16_256() {
; ALL-LABEL: ret_i16_256:
; ALL-DAG:       jr $ra
; ALL-DAG:       addiu $2, $zero, 256
  ret i16 256
}

define i16 @ret_i16_257() {
; ALL-LABEL: ret_i16_257:
; ALL-DAG:       jr $ra
; ALL-DAG:       addiu $2, $zero, 257
  ret i16 257
}

define i32 @ret_i32_257() {
; ALL-LABEL: ret_i32_257:
; ALL-DAG:       jr $ra
; ALL-DAG:       addiu $2, $zero, 257
  ret i32 257
}

define i32 @ret_i32_65536() {
; ALL-LABEL: ret_i32_65536:
; ALL-DAG:       jr $ra
; ALL-DAG:       lui $2, 1
  ret i32 65536
}

define i32 @ret_i32_65537() {
; ALL-LABEL: ret_i32_65537:
; ALL:           lui $[[T0:[0-9]+]], 1
; ALL-DAG:       jr $ra
; ALL-DAG:       ori $2, $[[T0]], 1
  ret i32 65537
}

define i64 @ret_i64_65537() {
; ALL-LABEL: ret_i64_65537:
; ALL:           lui $[[T0:[0-9]+]], 1

; GPR32-DAG:     ori $3, $[[T0]], 1
; GPR32-DAG:     addiu $2, $zero, 0

; GPR64-DAG:     daddiu $2, $[[T0]], 1

; ALL-DAG:       jr $ra
  ret i64 65537
}

define i64 @ret_i64_281479271677952() {
; ALL-LABEL: ret_i64_281479271677952:
; ALL-DAG:       lui $[[T0:[0-9]+]], 1

; GPR32-DAG:     ori $2, $[[T0]], 1
; GPR32-DAG:     addiu $3, $zero, 0

; GPR64-DAG:     daddiu $[[T1:[0-9]+]], $[[T0]], 1
; GPR64-DAG:     dsll $2, $[[T1]], 32

; ALL-DAG:       jr $ra
  ret i64 281479271677952
}

define i64 @ret_i64_281479271809026() {
; ALL-LABEL: ret_i64_281479271809026:
; GPR32-DAG:     lui $[[T0:[0-9]+]], 1
; GPR32-DAG:     lui $[[T1:[0-9]+]], 2
; GPR32-DAG:     ori $2, $[[T0]], 1
; GPR32-DAG:     ori $3, $[[T1]], 2

; GPR64-DAG:     ori  $[[T0:[0-9]+]], $zero, 32769
; GPR64-DAG:     dsll $[[T1:[0-9]+]], $[[T0]], 16
; GPR64-DAG:     daddiu $[[T0:[0-9]+]], $[[T0]], -32767
; GPR64-DAG:     dsll $[[T1:[0-9]+]], $[[T0]], 17
; GPR64-DAG:     daddiu $2, $[[T1]], 2

; ALL-DAG:       jr $ra
  ret i64 281479271809026
}

; TODO: f32
define float @ret_float_0x0() {
; ALL-LABEL: ret_float_0x0:

; NO-MTHC1-DAG:  mtc1 $zero, $f0

; MTHC1-DAG:     mtc1 $zero, $f0

; DMTC-DAG:      dmtc1 $zero, $f0

; ALL-DAG:       jr $ra
  ret float 0x0000000000000000
}

define float @ret_float_0x3() {
; ALL-LABEL: ret_float_0x3:

; Use a constant pool
; O32-DAG:       lwc1 $f0, %lo($CPI
; N64-DAG:       lwc1 $f0, %got_ofst($CPI

; ALL-DAG:       jr $ra

; float constants are written as double constants
  ret float 0x36b8000000000000
}

define double @ret_double_0x0() {
; ALL-LABEL: ret_double_0x0:

; NO-MTHC1-DAG:  mtc1 $zero, $f0
; NO-MTHC1-DAG:  mtc1 $zero, $f1

; MTHC1-DAG:     mtc1 $zero, $f0
; MTHC1-DAG:     mthc1 $zero, $f0

; DMTC-DAG:      dmtc1 $zero, $f0

; ALL-DAG:       jr $ra
  ret double 0x0000000000000000
}

define double @ret_double_0x3() {
; ALL-LABEL: ret_double_0x3:

; Use a constant pool
; O32-DAG:       ldc1 $f0, %lo($CPI
; N64-DAG:       ldc1 $f0, %got_ofst($CPI

; ALL-DAG:       jr $ra
  ret double 0x0000000000000003
}
