; RUN: llc -mtriple=x86_64-linux < %s | FileCheck %s

; CHECK: DW_AT_const_value
; CHECK-NEXT: 42

define i128 @__foo(i128 %a, i128 %b) nounwind {
entry:
  tail call void @llvm.dbg.value(metadata !0, i64 0, metadata !1, metadata !{metadata !"0x102"}), !dbg !11
  %add = add i128 %a, %b, !dbg !11
  ret i128 %add, !dbg !11
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!16}

!0 = metadata !{i128 42 }
!1 = metadata !{metadata !"0x100\00MAX\0029\000", metadata !2, metadata !4, metadata !8} ; [ DW_TAG_auto_variable ]
!2 = metadata !{metadata !"0xb\0026\000\000", metadata !13, metadata !3} ; [ DW_TAG_lexical_block ]
!3 = metadata !{metadata !"0x2e\00__foo\00__foo\00__foo\0026\000\001\000\006\000\000\0026", metadata !13, metadata !4, metadata !6, null, i128 (i128, i128)* @__foo, null, null, null} ; [ DW_TAG_subprogram ]
!4 = metadata !{metadata !"0x29", metadata !13} ; [ DW_TAG_file_type ]
!5 = metadata !{metadata !"0x11\001\00clang\001\00\000\00\000", metadata !13, metadata !15, metadata !15, metadata !12, null,  null} ; [ DW_TAG_compile_unit ]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !13, metadata !4, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !8, metadata !8}
!8 = metadata !{metadata !"0x16\00ti_int\0078\000\000\000\000", metadata !14, metadata !4, metadata !10} ; [ DW_TAG_typedef ]
!9 = metadata !{metadata !"0x29", metadata !14} ; [ DW_TAG_file_type ]
!10 = metadata !{metadata !"0x24\00\000\00128\00128\000\000\005", metadata !13, metadata !4} ; [ DW_TAG_base_type ]
!11 = metadata !{i32 29, i32 0, metadata !2, null}
!12 = metadata !{metadata !3}
!13 = metadata !{metadata !"foo.c", metadata !"/tmp"}
!14 = metadata !{metadata !"myint.h", metadata !"/tmp"}
!15 = metadata !{i32 0}
!16 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
