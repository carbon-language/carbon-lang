; RUN: opt -strip-debug < %s -S | FileCheck %s

; CHECK-NOT: llvm.dbg

@x = common global i32 0                          ; <i32*> [#uses=0]

define void @foo() nounwind readnone optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata !9, i64 0, metadata !5, metadata !{}), !dbg !10
  ret void, !dbg !11
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13}
!llvm.dbg.sp = !{!0}
!llvm.dbg.lv.foo = !{!5}
!llvm.dbg.gv = !{!8}

!0 = metadata !{metadata !"0x2e\00foo\00foo\00foo\002\000\001\000\006\000\001\000", metadata !12, metadata !1, metadata !3, null, void ()* @foo, null, null, null} ; [ DW_TAG_subprogram ]
!1 = metadata !{metadata !"0x29", metadata !12} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\001\00\000\00\000", metadata !12, metadata !4, metadata !4, null, null, null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !12, metadata !1, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{null}
!5 = metadata !{metadata !"0x100\00y\003\000", metadata !6, metadata !1, metadata !7} ; [ DW_TAG_auto_variable ]
!6 = metadata !{metadata !"0xb\002\000\000", metadata !12, metadata !0} ; [ DW_TAG_lexical_block ]
!7 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !12, metadata !1} ; [ DW_TAG_base_type ]
!8 = metadata !{metadata !"0x34\00x\00x\00\001\000\001", metadata !1, metadata !1, metadata !7, i32* @x} ; [ DW_TAG_variable ]
!9 = metadata !{i32 0}
!10 = metadata !{i32 3, i32 0, metadata !6, null}
!11 = metadata !{i32 4, i32 0, metadata !6, null}
!12 = metadata !{metadata !"b.c", metadata !"/tmp"}
!13 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
