; RUN: llc -O0 %s -mtriple=x86_64-apple-darwin -filetype=obj -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s
; RUN: llc -O0 %s -mtriple=x86_64-apple-darwin -filetype=obj -o %t -dwarf-version=3
; RUN: llvm-dwarfdump %t | FileCheck %s -check-prefix=DWARF3
; RUN: llc < %s -O0 -mtriple=x86_64-apple-macosx10.7 | FileCheck %s -check-prefix=ASM

; rdar://13067005
; CHECK: .debug_info contents:
; CHECK: DW_TAG_compile_unit
; CHECK: DW_AT_stmt_list [DW_FORM_sec_offset]   (0x00000000)
; CHECK: DW_AT_low_pc [DW_FORM_addr]            (0x0000000000000000)
; CHECK: DW_AT_high_pc [DW_FORM_data4]          (0x00000010)
; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_low_pc [DW_FORM_addr]            (0x0000000000000000)
; CHECK: DW_AT_high_pc [DW_FORM_data4]          (0x00000010)

; CHECK: DW_TAG_compile_unit
; CHECK: DW_AT_stmt_list [DW_FORM_sec_offset]   (0x0000003c)
; CHECK: DW_AT_low_pc [DW_FORM_addr]            (0x0000000000000010)
; CHECK: DW_AT_high_pc [DW_FORM_data4]          (0x00000009)
; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_low_pc [DW_FORM_addr]            (0x0000000000000010)
; CHECK: DW_AT_high_pc [DW_FORM_data4]          (0x00000009)


; CHECK: .debug_line contents:
; CHECK-NEXT: Line table prologue:
; CHECK-NEXT: total_length: 0x00000038
; CHECK: file_names[  1]    0 0x00000000 0x00000000 simple.c
; CHECK: Line table prologue:
; CHECK-NEXT: total_length: 0x00000039
; CHECK: file_names[  1]    0 0x00000000 0x00000000 simple2.c
; CHECK-NOT: file_names

; DWARF3: .debug_info contents:
; DWARF3: DW_TAG_compile_unit
; DWARF3: DW_AT_stmt_list [DW_FORM_data4]    (0x00000000)

; DWARF3: DW_TAG_compile_unit
; DWARF3: DW_AT_stmt_list [DW_FORM_data4]   (0x0000003c)


; DWARF3: .debug_line contents:
; DWARF3-NEXT: Line table prologue:
; DWARF3-NEXT: total_length: 0x00000038
; DWARF3: file_names[  1]    0 0x00000000 0x00000000 simple.c
; DWARF3: Line table prologue:
; DWARF3-NEXT: total_length: 0x00000039
; DWARF3: file_names[  1]    0 0x00000000 0x00000000 simple2.c
; DWARF3-NOT: file_names

; PR15408
; ASM: L__DWARF__debug_info_begin0:
; ASM: Lset3 = Lline_table_start0-Lsection_line ## DW_AT_stmt_list
; ASM-NEXT: .long   Lset3
; ASM: L__DWARF__debug_info_begin1:
; ASM: Lset13 = Lline_table_start0-Lsection_line ## DW_AT_stmt_list
; ASM-NEXT: .long   Lset13
define i32 @test(i32 %a) nounwind uwtable ssp {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %a.addr}, metadata !15, metadata !{metadata !"0x102"}), !dbg !16
  %0 = load i32* %a.addr, align 4, !dbg !17
  %call = call i32 @fn(i32 %0), !dbg !17
  ret i32 %call, !dbg !17
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define i32 @fn(i32 %a) nounwind uwtable ssp {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %a.addr}, metadata !19, metadata !{metadata !"0x102"}), !dbg !20
  %0 = load i32* %a.addr, align 4, !dbg !21
  ret i32 %0, !dbg !21
}

!llvm.dbg.cu = !{!0, !10}
!llvm.module.flags = !{!25}
!0 = metadata !{metadata !"0x11\0012\00clang version 3.3\000\00\000\00\001", metadata !23, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x2e\00test\00test\00\002\000\001\000\006\00256\000\003", metadata !23, metadata !6, metadata !7, null, i32 (i32)* @test, null, null, metadata !1} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 3] [test]
!6 = metadata !{metadata !"0x29", metadata !23} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !9}
!9 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{metadata !"0x11\0012\00clang version 3.3 (trunk 172862)\000\00\000\00\001", metadata !24, metadata !1, metadata !1, metadata !11, metadata !1,  metadata !1} ; [ DW_TAG_compile_unit ]
!11 = metadata !{metadata !13}
!13 = metadata !{metadata !"0x2e\00fn\00fn\00\001\000\001\000\006\00256\000\001", metadata !24, metadata !14, metadata !7, null, i32 (i32)* @fn, null, null, metadata !1} ; [ DW_TAG_subprogram ] [line 1] [def] [fn]
!14 = metadata !{metadata !"0x29", metadata !24} ; [ DW_TAG_file_type ]
!15 = metadata !{metadata !"0x101\00a\0016777218\000", metadata !5, metadata !6, metadata !9} ; [ DW_TAG_arg_variable ] [a] [line 2]
!16 = metadata !{i32 2, i32 0, metadata !5, null}
!17 = metadata !{i32 4, i32 0, metadata !18, null}
!18 = metadata !{metadata !"0xb\003\000\000", metadata !23, metadata !5} ; [ DW_TAG_lexical_block ]
!19 = metadata !{metadata !"0x101\00a\0016777217\000", metadata !13, metadata !14, metadata !9} ; [ DW_TAG_arg_variable ] [a] [line 1]
!20 = metadata !{i32 1, i32 0, metadata !13, null}
!21 = metadata !{i32 2, i32 0, metadata !22, null}
!22 = metadata !{metadata !"0xb\001\000\000", metadata !24, metadata !13} ; [ DW_TAG_lexical_block ]
!23 = metadata !{metadata !"simple.c", metadata !"/private/tmp"}
!24 = metadata !{metadata !"simple2.c", metadata !"/private/tmp"}
!25 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
