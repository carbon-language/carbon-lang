; RUN: llc -O0 -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s --check-prefix=LINUX
; RUN: llc -O0 -mtriple=x86_64-darwin < %s | FileCheck %s --check-prefix=DARWIN
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = common global i32 0, align 4
@yyyyyyyy = common global i32 0, align 4

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 720913, i32 0, i32 12, metadata !"hello.c", metadata !"/home/nlewycky", metadata !"clang version 3.1 (trunk 143048)", i1 true, i1 true, metadata !"", i32 0, metadata !1, metadata !1, metadata !1, metadata !3} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5, metadata !8}
!5 = metadata !{i32 720948, i32 0, null, metadata !"x", metadata !"x", metadata !"", metadata !6, i32 1, metadata !7, i32 0, i32 1, i32* @x} ; [ DW_TAG_variable ]
!6 = metadata !{i32 720937, metadata !"hello.c", metadata !"/home/nlewycky", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720932, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!8 = metadata !{i32 720948, i32 0, null, metadata !"yyyyyyyy", metadata !"yyyyyyyy", metadata !"", metadata !6, i32 2, metadata !7, i32 0, i32 1, i32* @yyyyyyyy} ; [ DW_TAG_variable ]

; 120 is ASCII 'x'. Verify that we use it directly as its name and don't emit
; a reference to the string pool.
; LINUX:        .byte   120                     # DW_AT_name
; DARWIN:       .byte   120                     ## DW_AT_name

; Verify that we refer to 'yyyyyyyy' with a relocation.
; LINUX:      .long   .Lstring{{[0-9]+}}      # DW_AT_name
; LINUX-NEXT: .long   39                      # DW_AT_type
; LINUX-NEXT: .byte   1                       # DW_AT_external
; LINUX-NEXT: .byte   1                       # DW_AT_decl_file
; LINUX-NEXT: .byte   2                       # DW_AT_decl_line
; LINUX-NEXT: .byte   9                       # DW_AT_location
; LINUX-NEXT: .byte   3
; LINUX-NEXT: .quad   yyyyyyyy

; Verify that we refer to 'yyyyyyyy' without a relocation.
; DARWIN: Lset[[N:[0-9]+]] = Lstring{{[0-9]+}}-Lsection_str   ## DW_AT_name
; DARWIN-NEXT:        .long   Lset[[N]]
; DARWIN-NEXT:        .long   39                      ## DW_AT_type
; DARWIN-NEXT:        .byte   1                       ## DW_AT_external
; DARWIN-NEXT:        .byte   1                       ## DW_AT_decl_file
; DARWIN-NEXT:        .byte   2                       ## DW_AT_decl_line
; DARWIN-NEXT:        .byte   9                       ## DW_AT_location
; DARWIN-NEXT:        .byte   3
; DARWIN-NEXT:        .quad   _yyyyyyyy


; Verify that "yyyyyyyy" ended up in the stringpool.
; LINUX: .section .debug_str,"MS",@progbits,1
; LINUX-NOT: .section
; LINUX: yyyyyyyy
; DARWIN: .section __DWARF,__debug_str,regular,debug
; DARWIN-NOT: .section
; DARWIN: yyyyyyyy
