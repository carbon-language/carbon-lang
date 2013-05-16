; RUN: llc -march=mipsel < %s | FileCheck %s

declare i8* @llvm.frameaddress(i32) nounwind readnone

define i8* @f() nounwind uwtable {
entry:
  %0 = call i8* @llvm.frameaddress(i32 0)
  ret i8* %0

; CHECK: .cfi_startproc
; CHECK: .cfi_def_cfa_offset 8
; CHECK: .cfi_offset 30, -4
; CHECK:   move    $fp, $sp
; CHECK: .cfi_def_cfa_register 30
; CHECK:   move    $2, $fp
; CHECK: .cfi_endproc
}
