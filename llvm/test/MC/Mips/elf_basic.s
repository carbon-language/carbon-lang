; RUN: llc -filetype=obj -mtriple mips-unknown-linux %s -o - | elf-dump --dump-section-data  | FileCheck -check-prefix=CHECK-BE %s
; RUN: llc -filetype=obj -mtriple mipsel-unknown-linux %s -o - | elf-dump --dump-section-data  | FileCheck -check-prefix=CHECK-LE %s

; Check that we produce the correct endian.

; CHECK-BE: ('e_indent[EI_DATA]', 0x02)
; CHECK-LE: ('e_indent[EI_DATA]', 0x01)
