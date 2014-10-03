; RUN: opt -S -simplifycfg < %s | FileCheck %s
; Radar 9342286
; Assign DebugLoc to trap instruction.
define void @foo() nounwind ssp {
; CHECK: call void @llvm.trap(), !dbg
  store i32 42, i32* null, !dbg !5
  ret void, !dbg !7
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10}
!llvm.dbg.sp = !{!0}

!0 = metadata !{metadata !"0x2e\00foo\00foo\00\003\000\001\000\006\000\000\000", metadata !8, metadata !1, metadata !3, null, void ()* @foo, null, null, null} ; [ DW_TAG_subprogram ] [line 3] [def] [scope 0] [foo]
!1 = metadata !{metadata !"0x29", metadata !8} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\0012\00Apple clang version 3.0 (tags/Apple/clang-206.1) (based on LLVM 3.0svn)\001\00\000\00\000", metadata !8, metadata !4, metadata !4, metadata !9, null, null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !8, metadata !1, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{null}
!5 = metadata !{i32 4, i32 2, metadata !6, null}
!6 = metadata !{metadata !"0xb\003\0012\000", metadata !8, metadata !0} ; [ DW_TAG_lexical_block ]
!7 = metadata !{i32 5, i32 1, metadata !6, null}
!8 = metadata !{metadata !"foo.c", metadata !"/private/tmp"}
!9 = metadata !{metadata !0}
!10 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
