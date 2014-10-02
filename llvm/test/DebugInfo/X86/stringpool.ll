; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s --check-prefix=LINUX
; RUN: llc -mtriple=x86_64-darwin < %s | FileCheck %s --check-prefix=DARWIN

@yyyy = common global i32 0, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.1 (trunk 143009)\001\00\000\00\000", metadata !8, metadata !1, metadata !1, metadata !1, metadata !3,  metadata !1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x34\00yyyy\00yyyy\00\001\000\001", null, metadata !6, metadata !7, i32* @yyyy, null} ; [ DW_TAG_variable ]
!6 = metadata !{metadata !"0x29", metadata !8} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!8 = metadata !{metadata !"z.c", metadata !"/home/nicholas"}

; Verify that "yyyy" ended up in the stringpool.
; LINUX: .section .debug_str,"MS",@progbits,1
; LINUX: yyyy
; DARWIN: .section __DWARF,__debug_str,regular,debug
; DARWIN: yyyy

; Verify that we refer to 'yyyy' with a relocation.
; LINUX:      .long   .Linfo_string3          # DW_AT_name
; LINUX-NEXT: .long   {{[0-9]+}}              # DW_AT_type
; LINUX-NEXT:                                 # DW_AT_external
; LINUX-NEXT: .byte   1                       # DW_AT_decl_file
; LINUX-NEXT: .byte   1                       # DW_AT_decl_line
; LINUX-NEXT: .byte   9                       # DW_AT_location
; LINUX-NEXT: .byte   3
; LINUX-NEXT: .quad   yyyy

; Verify that we refer to 'yyyy' without a relocation.
; DARWIN: Lset[[ID:[0-9]+]] = Linfo_string3-Linfo_string ## DW_AT_name
; DARWIN-NEXT:        .long   Lset[[ID]]
; DARWIN-NEXT:        .long   {{[0-9]+}}              ## DW_AT_type
; DARWIN-NEXT:                                        ## DW_AT_external
; DARWIN-NEXT:        .byte   1                       ## DW_AT_decl_file
; DARWIN-NEXT:        .byte   1                       ## DW_AT_decl_line
; DARWIN-NEXT:        .byte   9                       ## DW_AT_location
; DARWIN-NEXT:        .byte   3
; DARWIN-NEXT:        .quad   _yyyy
!9 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
