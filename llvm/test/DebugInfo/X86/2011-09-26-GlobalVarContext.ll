; RUN: llc -mtriple=x86_64-pc-linux-gnu %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; ModuleID = 'test.c'

@GLB = common global i32 0, align 4

define i32 @f() nounwind {
  %LOC = alloca i32, align 4
  call void @llvm.dbg.declare(metadata !{i32* %LOC}, metadata !15, metadata !{metadata !"0x102"}), !dbg !17
  %1 = load i32* @GLB, align 4, !dbg !18
  store i32 %1, i32* %LOC, align 4, !dbg !18
  %2 = load i32* @GLB, align 4, !dbg !19
  ret i32 %2, !dbg !19
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.0 (trunk)\000\00\000\00\000", metadata !20, metadata !1, metadata !1, metadata !3, metadata !12,  metadata !1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x2e\00f\00f\00\003\000\001\000\006\000\000\000", metadata !6, metadata !6, metadata !7, null, i32 ()* @f, null, null, null} ; [ DW_TAG_subprogram ] [line 3] [def] [scope 0] [f]
!6 = metadata !{metadata !"0x29", metadata !20} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!12 = metadata !{metadata !14}
!14 = metadata !{metadata !"0x34\00GLB\00GLB\00\001\000\001", null, metadata !6, metadata !9, i32* @GLB, null} ; [ DW_TAG_variable ]
!15 = metadata !{metadata !"0x100\00LOC\004\000", metadata !16, metadata !6, metadata !9} ; [ DW_TAG_auto_variable ]
!16 = metadata !{metadata !"0xb\003\009\000", metadata !20, metadata !5} ; [ DW_TAG_lexical_block ]
!17 = metadata !{i32 4, i32 9, metadata !16, null}
!18 = metadata !{i32 4, i32 23, metadata !16, null}
!19 = metadata !{i32 5, i32 5, metadata !16, null}
!20 = metadata !{metadata !"test.c", metadata !"/work/llvm/vanilla/test/DebugInfo"}

; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name [DW_FORM_strp]       ( .debug_str[0x{{[0-9a-f]*}}] = "GLB")
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_decl_file [DW_FORM_data1] ("/work/llvm/vanilla/test/DebugInfo{{[/\\]}}test.c")
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_decl_line [DW_FORM_data1] (1)

; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name [DW_FORM_strp]   ( .debug_str[0x{{[0-9a-f]*}}] = "LOC")
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_decl_file [DW_FORM_data1]     ("/work/llvm/vanilla/test/DebugInfo{{[/\\]}}test.c")
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_decl_line [DW_FORM_data1]     (4)

!21 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
