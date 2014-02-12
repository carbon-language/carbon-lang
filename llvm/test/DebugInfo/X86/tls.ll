; RUN: llc %s -o - -filetype=asm -O0 -mtriple=x86_64-unknown-linux-gnu \
; RUN:   | FileCheck --check-prefix=CHECK --check-prefix=SINGLE --check-prefix=SINGLE-64 %s

; RUN: llc %s -o - -filetype=asm -O0 -mtriple=i386-linux-gnu \
; RUN:   | FileCheck --check-prefix=CHECK --check-prefix=SINGLE --check-prefix=SINGLE-32 %s

; RUN: llc %s -o - -filetype=asm -O0 -mtriple=x86_64-unknown-linux-gnu -split-dwarf=Enable \
; RUN:   | FileCheck --check-prefix=CHECK --check-prefix=FISSION %s

; FIXME: add relocation and DWARF expression support to llvm-dwarfdump & use
; that here instead of raw assembly printing

; FISSION: .section    .debug_info.dwo,
; 3 bytes of data in this DW_FORM_block1 representation of the location of 'tls'
; FISSION: .byte 3{{ *}}# DW_AT_location
; DW_OP_const_index (0xfx == 252) to refer to the debug_addr table
; FISSION-NEXT: .byte 252
; an index of zero into the debug_addr table
; FISSION-NEXT: .byte 0

; SINGLE: .section     .debug_info,
; 10 bytes of data in this DW_FORM_block1 representation of the location of 'tls'
; SINGLE-64: .byte     10 # DW_AT_location
; DW_OP_const8u (0x0e == 14) of address
; SINGLE-64-NEXT: .byte        14
; SINGLE-64-NEXT: .quad tls@DTPOFF

; SINGLE-32: .byte     6 # DW_AT_location
; DW_OP_const4u (0x0e == 12) of address
; SINGLE-32-NEXT: .byte        12
; SINGLE-32-NEXT: .long tls@DTPOFF

; DW_OP_GNU_push_tls_address
; CHECK-NEXT: .byte 224

; check that the expected TLS address description is the first thing in the debug_addr section
; FISSION: .section    .debug_addr
; FISSION-NEXT: .quad  tls@DTPOFF

@tls = thread_local global i32 0, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !2, metadata !3, metadata !2, metadata !"tls.dwo"} ; [ DW_TAG_compile_unit ] [/tmp/tls.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"tls.cpp", metadata !"/tmp"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786484, i32 0, null, metadata !"tls", metadata !"tls", metadata !"", metadata !5, i32 1, metadata !6, i32 0, i32 1, i32* @tls, null} ; [ DW_TAG_variable ] [tls] [line 1] [def]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/tmp/tls.cpp]
!6 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!7 = metadata !{i32 2, metadata !"Dwarf Version", i32 3}
!8 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
