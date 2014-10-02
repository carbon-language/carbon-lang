; RUN: llc -O0 -mtriple=x86_64-apple-darwin10 < %s - | FileCheck %s
; Radar 8286101
; CHECK: .file   {{[0-9]+}} "<stdin>"

define i32 @foo() nounwind ssp {
entry:
  ret i32 42, !dbg !8
}

define i32 @bar() nounwind ssp {
entry:
  ret i32 21, !dbg !10
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!17}

!0 = metadata !{metadata !"0x2e\00foo\00foo\00foo\0053\000\001\000\006\000\000\000", metadata !14, metadata !1, metadata !3, null, i32 ()* @foo, null, null, null} ; [ DW_TAG_subprogram ]
!1 = metadata !{metadata !"0x29", metadata !14} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\0012\00clang version 2.9 (trunk 114084)\000\00\000\00\000", metadata !15, metadata !16, metadata !16, metadata !13, null, null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !14, metadata !1, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !14, metadata !1} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !"0x2e\00bar\00bar\00bar\004\000\001\000\006\000\000\000", metadata !15, metadata !7, metadata !3, null, i32 ()* @bar, null, null, null} ; [ DW_TAG_subprogram ]
!7 = metadata !{metadata !"0x29", metadata !15} ; [ DW_TAG_file_type ]
!8 = metadata !{i32 53, i32 13, metadata !9, null}
!9 = metadata !{metadata !"0xb\0053\0011\000", metadata !14, metadata !0} ; [ DW_TAG_lexical_block ]
!10 = metadata !{i32 4, i32 13, metadata !11, null}
!11 = metadata !{metadata !"0xb\004\0013\002", metadata !15, metadata !12} ; [ DW_TAG_lexical_block ]
!12 = metadata !{metadata !"0xb\004\0011\001", metadata !15, metadata !6} ; [ DW_TAG_lexical_block ]
!13 = metadata !{metadata !0, metadata !6}
!14 = metadata !{metadata !"", metadata !"/private/tmp"}
!15 = metadata !{metadata !"bug.c", metadata !"/private/tmp"}
!16 = metadata !{i32 0}
!17 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
