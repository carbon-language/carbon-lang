; RUN: llc < %s -o /dev/null
; PR 5942
define i8* @foo() nounwind {
entry:
  %0 = load i32* undef, align 4, !dbg !0          ; <i32> [#uses=1]
  %1 = inttoptr i32 %0 to i8*, !dbg !0            ; <i8*> [#uses=1]
  ret i8* %1, !dbg !10

}

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!14}

!0 = metadata !{i32 571, i32 3, metadata !1, null}
!1 = metadata !{metadata !"0xb\001\001\000", metadata !11, metadata !2}; [DW_TAG_lexical_block ]
!2 = metadata !{metadata !"0x2e\00foo\00foo\00foo\00561\000\001\000\006\000\000\000", i32 0, metadata !3, metadata !4, null, null, null, null, null}; [DW_TAG_subprogram ]
!3 = metadata !{metadata !"0x11\0012\00clang 1.1\001\00\000\00\000", metadata !11, metadata !12, metadata !12, metadata !13, null, null}; [DW_TAG_compile_unit ]
!4 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", null, metadata !3, null, metadata !5, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = metadata !{metadata !6}
!6 = metadata !{metadata !"0x24\00char\000\008\008\000\000\006", null, metadata !3} ; [ DW_TAG_base_type ]
!10 = metadata !{i32 588, i32 1, metadata !2, null}
!11 = metadata !{metadata !"hashtab.c", metadata !"/usr/src/gnu/usr.bin/cc/cc_tools/../../../../contrib/gcclibs/libiberty"}
!12 = metadata !{i32 0}
!13 = metadata !{metadata !2}
!14 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
