; This file is used by 2011-08-04-Metadata.ll, so it doesn't actually do anything itself
;
; RUN: true


target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

@x = internal global i32 0, align 4

define void @bar() nounwind uwtable ssp {
entry:
  store i32 1, i32* @x, align 4, !dbg !7
  ret void, !dbg !7
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11}
!llvm.dbg.sp = !{!1}
!llvm.dbg.gv = !{!5}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.0 ()\001\00\000\00\000", metadata !9, metadata !4, metadata !4, metadata !10, null, null} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !"0x2e\00bar\00bar\00\002\000\001\000\006\000\000\000", metadata !9, metadata !2, metadata !3, null, void ()* @bar, null, null, null} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 0] [bar]
!2 = metadata !{metadata !"0x29", metadata !9} ; [ DW_TAG_file_type ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !9, metadata !2, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{null}
!5 = metadata !{metadata !"0x34\00x\00x\00\001\001\001", metadata !0, metadata !2, metadata !6, i32* @x} ; [ DW_TAG_variable ]
!6 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, metadata !0} ; [ DW_TAG_base_type ]
!7 = metadata !{i32 2, i32 14, metadata !8, null}
!8 = metadata !{metadata !"0xb\002\0012\000", metadata !9, metadata !1} ; [ DW_TAG_lexical_block ]
!9 = metadata !{metadata !"/tmp/two.c", metadata !"/Volumes/Lalgate/Slate/D"}
!10 = metadata !{metadata !1}
!11 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
