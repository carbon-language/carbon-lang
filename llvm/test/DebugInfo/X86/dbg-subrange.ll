; RUN: llc -O0 < %s | FileCheck %s
; Radar 10464995
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.2"

@s = common global [4294967296 x i8] zeroinitializer, align 16
; CHECK: .quad 4294967296 ## DW_AT_count

define void @bar() nounwind uwtable ssp {
entry:
  store i8 97, i8* getelementptr inbounds ([4294967296 x i8]* @s, i32 0, i64 0), align 1, !dbg !18
  ret void, !dbg !20
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.1 (trunk 144833)\000\00\000\00\000", metadata !21, metadata !1, metadata !1, metadata !3, metadata !11,  metadata !1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x2e\00bar\00bar\00\004\000\001\000\006\00256\000\000", metadata !21, metadata !6, metadata !7, null, void ()* @bar, null, null, null} ; [ DW_TAG_subprogram ] [line 4] [def] [scope 0] [bar]
!6 = metadata !{metadata !"0x29", metadata !21} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null}
!11 = metadata !{metadata !13}
!13 = metadata !{metadata !"0x34\00s\00s\00\002\000\001", null, metadata !6, metadata !14, [4294967296 x i8]* @s, null} ; [ DW_TAG_variable ]
!14 = metadata !{metadata !"0x1\00\000\0034359738368\008\000\000", null, null, metadata !15, metadata !16, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 34359738368, align 8, offset 0] [from char]
!15 = metadata !{metadata !"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ]
!16 = metadata !{metadata !17}
!17 = metadata !{metadata !"0x21\000\004294967296"} ; [ DW_TAG_subrange_type ]
!18 = metadata !{i32 5, i32 3, metadata !19, null}
!19 = metadata !{metadata !"0xb\004\001\000", metadata !21, metadata !5} ; [ DW_TAG_lexical_block ]
!20 = metadata !{i32 6, i32 1, metadata !19, null}
!21 = metadata !{metadata !"small.c", metadata !"/private/tmp"}
!22 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
