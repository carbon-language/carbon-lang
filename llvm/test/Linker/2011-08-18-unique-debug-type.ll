; RUN: llvm-link %s %p/2011-08-18-unique-debug-type2.ll -S -o - | FileCheck %s
; Test to check only one MDNode for "int" after linking.
; CHECK: !"0x24\00int\00{{.*}}"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

define i32 @foo() nounwind uwtable ssp {
entry:
  ret i32 1, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.0 (trunk 137954)\001\00\000\00\000", metadata !12, metadata !2, metadata !2, metadata !3, metadata !2, null} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x2e\00foo\00foo\00\001\000\001\000\006\000\000\000", metadata !12, metadata !6, metadata !7, null, i32 ()* @foo, null, null, null} ; [ DW_TAG_subprogram ] [line 1] [def] [scope 0] [foo]
!6 = metadata !{metadata !"0x29", metadata !12} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !12, metadata !6, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!10 = metadata !{i32 1, i32 13, metadata !11, null}
!11 = metadata !{metadata !"0xb\001\0011\000", metadata !12, metadata !5} ; [ DW_TAG_lexical_block ]
!12 = metadata !{metadata !"one.c", metadata !"/private/tmp"}
!13 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
