; RUN: llc < %s -o /dev/null

define void @baz(i32 %i) nounwind ssp {
entry:
  call void @llvm.dbg.declare(metadata !0, metadata !1, metadata !{metadata !"0x102"}), !dbg !0
  ret void, !dbg !0
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!22}

!0 = metadata !{{ [0 x i8] }** undef}
!1 = metadata !{metadata !"0x100\00x\0011\000", metadata !2, metadata !4, metadata !9} ; [ DW_TAG_auto_variable ]
!2 = metadata !{metadata !"0xb\008\000\000", metadata !20, metadata !3} ; [ DW_TAG_lexical_block ]
!3 = metadata !{metadata !"0x2e\00baz\00baz\00baz\008\001\001\000\006\000\000\000", metadata !20, null, metadata !6, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!4 = metadata !{metadata !"0x29", metadata !20} ; [ DW_TAG_file_type ]
!5 = metadata !{metadata !"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\001\00\000\00\000", metadata !20, metadata !21, metadata !21, null, null, null} ; [ DW_TAG_compile_unit ]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !20, metadata !4, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null, metadata !8}
!8 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !20, metadata !4} ; [ DW_TAG_base_type ]
!9 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", metadata !20, metadata !4, metadata !10} ; [ DW_TAG_pointer_type ]
!10 = metadata !{metadata !"0x13\00\0011\008\008\000\000\000", metadata !20, metadata !3, null, metadata !11, null, null, null} ; [ DW_TAG_structure_type ] [line 11, size 8, align 8, offset 0] [def] [from ]
!11 = metadata !{metadata !12}
!12 = metadata !{metadata !"0xd\00b\0011\008\008\000\000", metadata !20, metadata !10, metadata !13} ; [ DW_TAG_member ]
!13 = metadata !{metadata !"0x16\00A\0011\000\000\000\000", metadata !20, metadata !3, metadata !14} ; [ DW_TAG_typedef ]
!14 = metadata !{metadata !"0x1\00\000\008\008\000\000", metadata !20, metadata !4, metadata !15, metadata !16, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 8, align 8, offset 0] [from char]
!15 = metadata !{metadata !"0x24\00char\000\008\008\000\000\006", metadata !20, metadata !4} ; [ DW_TAG_base_type ]
!16 = metadata !{metadata !17}
!17 = metadata !{metadata !"0x21\000\001"}        ; [ DW_TAG_subrange_type ]
!18 = metadata !{metadata !"llvm.mdnode.fwdref.19"}
!19 = metadata !{metadata !"llvm.mdnode.fwdref.23"}
!20 = metadata !{metadata !"2007-12-VarArrayDebug.c", metadata !"/Users/sabre/llvm/test/FrontendC/"}
!21 = metadata !{i32 0}
!22 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
