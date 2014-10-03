; RUN: opt -simplifycfg -S < %s | FileCheck %s

define i32 @foo(i32 %i) nounwind ssp {
  call void @llvm.dbg.value(metadata !{i32 %i}, i64 0, metadata !6, metadata !{}), !dbg !7
  call void @llvm.dbg.value(metadata !8, i64 0, metadata !9, metadata !{}), !dbg !11
  %1 = icmp ne i32 %i, 0, !dbg !12
;CHECK: call i32 (...)* @bar()
;CHECK-NEXT: llvm.dbg.value
  br i1 %1, label %2, label %4, !dbg !12

; <label>:2                                       ; preds = %0
  %3 = call i32 (...)* @bar(), !dbg !13
  call void @llvm.dbg.value(metadata !{i32 %3}, i64 0, metadata !9, metadata !{}), !dbg !13
  br label %6, !dbg !15

; <label>:4                                       ; preds = %0
  %5 = call i32 (...)* @bar(), !dbg !16
  call void @llvm.dbg.value(metadata !{i32 %5}, i64 0, metadata !9, metadata !{}), !dbg !16
  br label %6, !dbg !18

; <label>:6                                       ; preds = %4, %2
  %k.0 = phi i32 [ %3, %2 ], [ %5, %4 ]
  ret i32 %k.0, !dbg !19
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i32 @bar(...)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.module.flags = !{!21}
!llvm.dbg.sp = !{!0}

!0 = metadata !{metadata !"0x2e\00foo\00foo\00\002\000\001\000\006\00256\000\000", metadata !20, metadata !1, metadata !3, null, i32 (i32)* @foo, null, null, null} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 0] [foo]
!1 = metadata !{metadata !"0x29", metadata !20} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\0012\00clang\001\00\000\00\000", metadata !20, metadata !8, metadata !8, null, null, null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !20, metadata !1, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, metadata !2} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !"0x101\00i\0016777218\000", metadata !0, metadata !1, metadata !5} ; [ DW_TAG_arg_variable ]
!7 = metadata !{i32 2, i32 13, metadata !0, null}
!8 = metadata !{i32 0}
!9 = metadata !{metadata !"0x100\00k\003\000", metadata !10, metadata !1, metadata !5} ; [ DW_TAG_auto_variable ]
!10 = metadata !{metadata !"0xb\002\0016\000", metadata !20, metadata !0} ; [ DW_TAG_lexical_block ]
!11 = metadata !{i32 3, i32 12, metadata !10, null}
!12 = metadata !{i32 4, i32 3, metadata !10, null}
!13 = metadata !{i32 5, i32 5, metadata !14, null}
!14 = metadata !{metadata !"0xb\004\0010\001", metadata !20, metadata !10} ; [ DW_TAG_lexical_block ]
!15 = metadata !{i32 6, i32 3, metadata !14, null}
!16 = metadata !{i32 7, i32 5, metadata !17, null}
!17 = metadata !{metadata !"0xb\006\0010\002", metadata !20, metadata !10} ; [ DW_TAG_lexical_block ]
!18 = metadata !{i32 8, i32 3, metadata !17, null}
!19 = metadata !{i32 9, i32 3, metadata !10, null}
!20 = metadata !{metadata !"b.c", metadata !"/private/tmp"}
!21 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
