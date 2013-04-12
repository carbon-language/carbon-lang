; Codegen version to check for ELF header flags.
;
; RUN: llc %s -mtriple=thumbv7-linux-gnueabi -relocation-model=pic \
; RUN: -filetype=obj -o - | llvm-readobj -h | \
; RUN: FileCheck %s

define void @bar() nounwind {
entry:
  ret void
}

; For now the only e_flag set is EF_ARM_EABI_VER5
; CHECK: ElfHeader {
; CHECK:   Flags [ (0x5000000)
