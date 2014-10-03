; RUN: opt -instcombine -S < %s | FileCheck %s


@.str = private constant [3 x i8] c"%c\00"

define void @foo() nounwind ssp {
;CHECK: call i32 @putchar{{.+}} !dbg
  %1 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([3 x i8]* @.str, i32 0, i32 0), i32 97), !dbg !5
  ret void, !dbg !7
}

declare i32 @printf(i8*, ...)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10}
!llvm.dbg.sp = !{!0}

!0 = metadata !{metadata !"0x2e\00foo\00foo\00\004\000\001\000\006\000\000\000", metadata !8, metadata !1, metadata !3, null, void ()* @foo, null, null, null} ; [ DW_TAG_subprogram ]
!1 = metadata !{metadata !"0x29", metadata !8} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\0012\00clang\001\00\000\00\000", metadata !8, metadata !4, metadata !4, metadata !9, null, null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !8, metadata !1, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{null}
!5 = metadata !{i32 5, i32 2, metadata !6, null}
!6 = metadata !{metadata !"0xb\004\0012\000", metadata !8, metadata !0} ; [ DW_TAG_lexical_block ]
!7 = metadata !{i32 6, i32 1, metadata !6, null}
!8 = metadata !{metadata !"m.c", metadata !"/private/tmp"}
!9 = metadata !{metadata !0}
!10 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
