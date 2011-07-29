; RUN: opt -reassociate -S < %s | FileCheck %s
; PR 10176
define i64 @foo(i64 %a, i64 %b, i64 %c) nounwind uwtable readnone ssp {
entry:
  tail call void @llvm.dbg.value(metadata !{i64 %a}, i64 0, metadata !6), !dbg !11
  tail call void @llvm.dbg.value(metadata !{i64 %b}, i64 0, metadata !7), !dbg !12
  tail call void @llvm.dbg.value(metadata !{i64 %c}, i64 0, metadata !8), !dbg !13
  %add = add nsw i64 %c, %b, !dbg !14
;CHECK-NOT:   call void @llvm.dbg.value(metadata !{i64 %add}
  tail call void @llvm.dbg.value(metadata !{i64 %add}, i64 0, metadata !9), !dbg !14
  %add4 = add nsw i64 %add, %a, !dbg !15
  ret i64 %add4, !dbg !15
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.dbg.sp = !{!1}
!llvm.dbg.lv.foo = !{!6, !7, !8, !9}

!0 = metadata !{i32 655377, i32 0, i32 12, metadata !"dan.c", metadata !"/private/tmp", metadata !"clang version 3.0 (trunk 136263)", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 655406, i32 0, metadata !2, metadata !"foo", metadata !"foo", metadata !"", metadata !2, i32 1, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, i64 (i64, i64, i64)* @foo, null, null} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 655401, metadata !"dan.c", metadata !"/private/tmp", metadata !0} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 655381, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 655396, metadata !0, metadata !"long int", null, i32 0, i64 64, i64 64, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 655617, metadata !1, metadata !"a", metadata !2, i32 16777217, metadata !5, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!7 = metadata !{i32 655617, metadata !1, metadata !"b", metadata !2, i32 33554433, metadata !5, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!8 = metadata !{i32 655617, metadata !1, metadata !"c", metadata !2, i32 50331649, metadata !5, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!9 = metadata !{i32 655616, metadata !10, metadata !"d", metadata !2, i32 2, metadata !5, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!10 = metadata !{i32 655371, metadata !1, i32 1, i32 34, metadata !2, i32 0} ; [ DW_TAG_lexical_block ]
!11 = metadata !{i32 1, i32 15, metadata !1, null}
!12 = metadata !{i32 1, i32 23, metadata !1, null}
!13 = metadata !{i32 1, i32 31, metadata !1, null}
!14 = metadata !{i32 2, i32 17, metadata !10, null}
!15 = metadata !{i32 3, i32 3, metadata !10, null}
