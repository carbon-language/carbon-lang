; RUN: llc -mtriple=arm64-apple-ios %s -o - | FileCheck %s

; All global symbols must be at-most linker-private for AArch64 because we don't
; use section-relative relocations in MachO.

define i8* @private_sym() {
; CHECK-LABEL: private_sym:
; CHECK:     adrp [[HIBITS:x[0-9]+]], l_var@PAGE
; CHECK:     add x0, [[HIBITS]], l_var@PAGEOFF

  ret i8* getelementptr([2 x i8], [2 x i8]* @var, i32 0, i32 0)
}

; CHECK:     .section __TEXT,__cstring
; CHECK: l_var:
; CHECK:    .asciz "\002"
@var = private unnamed_addr constant [2 x i8] [i8 2, i8 0]
