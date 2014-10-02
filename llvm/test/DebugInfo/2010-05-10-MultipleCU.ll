; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Check that two compile units are generated

; CHECK: Compile Unit:
; CHECK: Compile Unit:

define i32 @foo() nounwind readnone ssp {
return:
  ret i32 42, !dbg !0
}

define i32 @bar() nounwind readnone ssp {
return:
  ret i32 21, !dbg !8
}

!llvm.dbg.cu = !{!4, !12}
!llvm.module.flags = !{!21}
!16 = metadata !{metadata !2}
!17 = metadata !{metadata !10}

!0 = metadata !{i32 3, i32 0, metadata !1, null}
!1 = metadata !{metadata !"0xb\002\000\000", metadata !18, metadata !2} ; [ DW_TAG_lexical_block ]
!2 = metadata !{metadata !"0x2e\00foo\00foo\00foo\002\000\001\000\006\000\000\000", metadata !18, metadata !3, metadata !5, null, i32 ()* @foo, null, null, null} ; [ DW_TAG_subprogram ]
!3 = metadata !{metadata !"0x29", metadata !18} ; [ DW_TAG_file_type ]
!4 = metadata !{metadata !"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\000\00\000\00\000", metadata !18, metadata !19, metadata !19, metadata !16, null, null} ; [ DW_TAG_compile_unit ]
!5 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !18, metadata !3, null, metadata !6, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!6 = metadata !{metadata !7}
!7 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !18, metadata !3} ; [ DW_TAG_base_type ]
!8 = metadata !{i32 3, i32 0, metadata !9, null}
!9 = metadata !{metadata !"0xb\002\000\000", metadata !20, metadata !10} ; [ DW_TAG_lexical_block ]
!10 = metadata !{metadata !"0x2e\00bar\00bar\00bar\002\000\001\000\006\000\000\000", metadata !20, metadata !11, metadata !13, null, i32 ()* @bar, null, null, null} ; [ DW_TAG_subprogram ]
!11 = metadata !{metadata !"0x29", metadata !20} ; [ DW_TAG_file_type ]
!12 = metadata !{metadata !"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\000\00\000\00\000", metadata !20, metadata !19, metadata !19, metadata !17, null, null} ; [ DW_TAG_compile_unit ]
!13 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !20, metadata !11, null, metadata !14, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = metadata !{metadata !15}
!15 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !20, metadata !11} ; [ DW_TAG_base_type ]
!18 = metadata !{metadata !"a.c", metadata !"/tmp/"}
!19 = metadata !{i32 0}
!20 = metadata !{metadata !"b.c", metadata !"/tmp/"}
!21 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
