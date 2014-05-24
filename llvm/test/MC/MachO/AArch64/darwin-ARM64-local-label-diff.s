; RUN: llvm-mc -triple arm64-apple-darwin -filetype=obj -o - < %s | macho-dump -dump-section-data | FileCheck %s
; rdar://13028719

 .globl context_save0
 .align 6
Lcontext_save0:
context_save0:
 .fill 2, 8, 5
Lcontext_save0_end:
Lcontext_save0_size: .quad (Lcontext_save0_end - Lcontext_save0)

 .align 6
Lcontext_save1:
 .fill 2, 8, 0
Lcontext_save1_end:
Lcontext_save1_size: .quad (Lcontext_save1_end - Lcontext_save1)

Llockup_release:
 .quad 0

; CHECK:  ('_section_data', '05000000 00000000 05000000 00000000 10000000 00000000 1f2003d5 1f2003d5 1f2003d5 1f2003d5 1f2003d5 1f2003d5 1f2003d5 1f2003d5 1f2003d5 1f2003d5 00000000 00000000 00000000 00000000 10000000 00000000 00000000 00000000')
