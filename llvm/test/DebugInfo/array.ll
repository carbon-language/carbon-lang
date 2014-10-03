; RUN: %llc_dwarf -O0 < %s | FileCheck %s
; Do not emit AT_upper_bound for an unbounded array.
; radar 9241695
define i32 @main() nounwind ssp {
entry:
  %retval = alloca i32, align 4
  %a = alloca [0 x i32], align 4
  store i32 0, i32* %retval
  call void @llvm.dbg.declare(metadata !{[0 x i32]* %a}, metadata !6, metadata !{metadata !"0x102"}), !dbg !11
  ret i32 0, !dbg !12
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!16}

!0 = metadata !{metadata !"0x2e\00main\00main\00\003\000\001\000\006\000\000\003", metadata !14, metadata !1, metadata !3, null, i32 ()* @main, null, null, null} ; [ DW_TAG_subprogram ] [line 3] [def] [main]
!1 = metadata !{metadata !"0x29", metadata !14} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\0012\00clang version 3.0 (trunk 129138)\000\00\000\00\000", metadata !14, metadata !15, metadata !15, metadata !13, null,  null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !14, metadata !1, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, metadata !2} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !"0x100\00a\004\000", metadata !7, metadata !1, metadata !8} ; [ DW_TAG_auto_variable ]
!7 = metadata !{metadata !"0xb\003\0012\000", metadata !14, metadata !0} ; [ DW_TAG_lexical_block ]
!8 = metadata !{metadata !"0x1\00\000\000\0032\000\000", metadata !14, metadata !2, metadata !5, metadata !9, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 0, align 32, offset 0] [from int]
!9 = metadata !{metadata !10}
;CHECK: DW_TAG_subrange_type
;CHECK-NEXT: DW_AT_type
;CHECK-NOT: DW_AT_lower_bound
;CHECK-NOT: DW_AT_upper_bound
;CHECK-NEXT: End Of Children Mark
!10 = metadata !{metadata !"0x21\000\00-1"}        ; [ DW_TAG_subrange_type ]
!11 = metadata !{i32 4, i32 7, metadata !7, null}
!12 = metadata !{i32 5, i32 3, metadata !7, null}
!13 = metadata !{metadata !0}
!14 = metadata !{metadata !"array.c", metadata !"/private/tmp"}
!15 = metadata !{i32 0}
!16 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
