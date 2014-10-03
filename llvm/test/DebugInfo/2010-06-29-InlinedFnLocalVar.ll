; RUN: %llc_dwarf -O2 %s -o - | FileCheck %s
; Check struct X for dead variable xyz from inlined function foo.

; CHECK:	DW_TAG_structure_type
; CHECK-NEXT:	info_string
 

@i = common global i32 0                          ; <i32*> [#uses=2]

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

define i32 @bar() nounwind ssp {
entry:
  %0 = load i32* @i, align 4, !dbg !17            ; <i32> [#uses=2]
  tail call void @llvm.dbg.value(metadata !{i32 %0}, i64 0, metadata !9, metadata !{metadata !"0x102"}), !dbg !19
  tail call void @llvm.dbg.declare(metadata !29, metadata !10, metadata !{metadata !"0x102"}), !dbg !21
  %1 = mul nsw i32 %0, %0, !dbg !22               ; <i32> [#uses=2]
  store i32 %1, i32* @i, align 4, !dbg !17
  ret i32 %1, !dbg !23
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!28}

!0 = metadata !{metadata !"0x2e\00foo\00foo\00\009\001\001\000\006\000\001\009", metadata !27, metadata !1, metadata !3, null, null, null, null, metadata !24} ; [ DW_TAG_subprogram ]
!1 = metadata !{metadata !"0x29", metadata !27} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\001\00\000\00\000", metadata !27, metadata !20, metadata !20, metadata !25, metadata !26,  metadata !20} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !27, metadata !1, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5, metadata !5}
!5 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !27, metadata !1} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !"0x2e\00bar\00bar\00bar\0014\000\001\000\006\000\001\000", metadata !27, metadata !1, metadata !7, null, i32 ()* @bar, null, null, null} ; [ DW_TAG_subprogram ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !27, metadata !1, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !5}
!9 = metadata !{metadata !"0x101\00j\009\000", metadata !0, metadata !1, metadata !5} ; [ DW_TAG_arg_variable ]
!10 = metadata !{metadata !"0x100\00xyz\0010\000", metadata !11, metadata !1, metadata !12} ; [ DW_TAG_auto_variable ]
!11 = metadata !{metadata !"0xb\009\000\000", metadata !1, metadata !0} ; [ DW_TAG_lexical_block ]
!12 = metadata !{metadata !"0x13\00X\0010\0064\0032\000\000\000", metadata !27, metadata !0, null, metadata !13, null, null, null} ; [ DW_TAG_structure_type ] [X] [line 10, size 64, align 32, offset 0] [def] [from ]
!13 = metadata !{metadata !14, metadata !15}
!14 = metadata !{metadata !"0xd\00a\0010\0032\0032\000\000", metadata !27, metadata !12, metadata !5} ; [ DW_TAG_member ]
!15 = metadata !{metadata !"0xd\00b\0010\0032\0032\0032\000", metadata !27, metadata !12, metadata !5} ; [ DW_TAG_member ]
!16 = metadata !{metadata !"0x34\00i\00i\00\005\000\001", metadata !1, metadata !1, metadata !5, i32* @i, null} ; [ DW_TAG_variable ]
!17 = metadata !{i32 15, i32 0, metadata !18, null}
!18 = metadata !{metadata !"0xb\0014\000\001", metadata !1, metadata !6} ; [ DW_TAG_lexical_block ]
!19 = metadata !{i32 9, i32 0, metadata !0, metadata !17}
!20 = metadata !{}
!21 = metadata !{i32 9, i32 0, metadata !11, metadata !17}
!22 = metadata !{i32 11, i32 0, metadata !11, metadata !17}
!23 = metadata !{i32 16, i32 0, metadata !18, null}
!24 = metadata !{metadata !9, metadata !10}
!25 = metadata !{metadata !0, metadata !6}
!26 = metadata !{metadata !16}
!27 = metadata !{metadata !"bar.c", metadata !"/tmp/"}
!28 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!29 = metadata !{null}
