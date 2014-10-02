; RUN: llvm-link %s %p/2011-08-04-DebugLoc2.ll -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s
; Test if DebugLoc is mapped properly or not.


;CHECK-NOT: metadata !{i32 589870, i32 0, metadata !{{[0-9]+}}, metadata !"bar", metadata !"bar", metadata !"", metadata !{{[0-9]+}}, i32 1, metadata !{{[0-9]+}}, i1 false, i1 true, i32 0, i32 0, i32 0, i32 0, i1 false, null, null, null}


target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

define i32 @foo() nounwind ssp {
  ret i32 42, !dbg !6
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11}
!llvm.dbg.sp = !{!1}

!0 = metadata !{metadata !"0x11\0012\00Apple clang version 3.0 (tags/Apple/clang-209.11) (based on LLVM 3.0svn)\001\00\000\00\000", metadata !8, metadata !9, metadata !9, metadata !10, null, null} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !"0x2e\00foo\00foo\00\002\000\001\000\006\000\000\000", metadata !8, metadata !2, metadata !3, null, i32 ()* @foo, null, null, null} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 0] [foo]
!2 = metadata !{metadata !"0x29", metadata !8} ; [ DW_TAG_file_type ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !8, metadata !2, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, metadata !0} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 2, i32 13, metadata !7, null}
!7 = metadata !{metadata !"0xb\002\0011\000", metadata !8, metadata !1} ; [ DW_TAG_lexical_block ]
!8 = metadata !{metadata !"a.c", metadata !"/private/tmp"}
!9 = metadata !{i32 0}
!10 = metadata !{metadata !1}
!11 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
