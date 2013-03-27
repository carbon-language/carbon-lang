; RUN: llc -mtriple=x86_64-apple-darwin %s -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
; RUN: llc -mtriple=x86_64-apple-darwin -regalloc=basic %s -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

;CHECK: DW_TAG_inlined_subroutine [12]
;CHECK-NEXT: DW_AT_abstract_origin
;CHECK-NEXT: DW_AT_low_pc
;CHECK-NEXT: DW_AT_high_pc
;CHECK-NEXT: DW_AT_call_file
;CHECK-NEXT: DW_AT_call_line

;CHECK: DW_TAG_formal_parameter [9]
;CHECK-NEXT: DW_AT_name [DW_FORM_strp] ( .debug_str[0x00000055] = "sp")

%struct.S1 = type { float*, i32 }

@p = common global %struct.S1 zeroinitializer, align 8

define i32 @foo(%struct.S1* nocapture %sp, i32 %nums) nounwind optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata !{%struct.S1* %sp}, i64 0, metadata !9), !dbg !20
  tail call void @llvm.dbg.value(metadata !{i32 %nums}, i64 0, metadata !18), !dbg !21
  %tmp2 = getelementptr inbounds %struct.S1* %sp, i64 0, i32 1, !dbg !22
  store i32 %nums, i32* %tmp2, align 4, !dbg !22, !tbaa !24
  %call = tail call float* @bar(i32 %nums) nounwind optsize, !dbg !27
  %tmp5 = getelementptr inbounds %struct.S1* %sp, i64 0, i32 0, !dbg !27
  store float* %call, float** %tmp5, align 8, !dbg !27, !tbaa !28
  %cmp = icmp ne float* %call, null, !dbg !29
  %cond = zext i1 %cmp to i32, !dbg !29
  ret i32 %cond, !dbg !29
}

declare float* @bar(i32) optsize

define void @foobar() nounwind optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata !30, i64 0, metadata !9) nounwind, !dbg !31
  tail call void @llvm.dbg.value(metadata !34, i64 0, metadata !18) nounwind, !dbg !35
  store i32 1, i32* getelementptr inbounds (%struct.S1* @p, i64 0, i32 1), align 8, !dbg !36, !tbaa !24
  %call.i = tail call float* @bar(i32 1) nounwind optsize, !dbg !37
  store float* %call.i, float** getelementptr inbounds (%struct.S1* @p, i64 0, i32 0), align 8, !dbg !37, !tbaa !28
  ret void, !dbg !38
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}

!0 = metadata !{i32 786478, metadata !1, null, metadata !1, metadata !"foo", metadata !"foo", metadata !"", i32 8, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, i32 (%struct.S1*, i32)* @foo, null, null, metadata !41, i32 8} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 786473, metadata !42, null} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 786449, metadata !42, null, i32 12, metadata !"clang version 2.9 (trunk 125693)", i1 true, metadata !"", i32 0, null, null, metadata !39, metadata !40, null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 786453, metadata !42, null, metadata !1, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786468, null, null, metadata !2, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 786478, metadata !1, null, metadata !1, metadata !"foobar", metadata !"foobar", metadata !"", i32 15, metadata !7, i1 false, i1 true, i32 0, i32 0, i32 0, i32 0, i1 true, void ()* @foobar} ; [ DW_TAG_subprogram ]
!7 = metadata !{i32 786453, metadata !42, null, metadata !1, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{null}
!9 = metadata !{i32 786689, metadata !0, metadata !"sp", metadata !1, i32 7, metadata !10, i32 0, metadata !32} ; [ DW_TAG_arg_variable ]
!10 = metadata !{i32 786447, null, null, metadata !2, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !11} ; [ DW_TAG_pointer_type ]
!11 = metadata !{i32 786454, metadata !42, null, metadata !2, metadata !"S1", i32 4, i64 0, i64 0, i64 0, i32 0, metadata !12} ; [ DW_TAG_typedef ]
!12 = metadata !{i32 786451, metadata !42, null, metadata !2, metadata !"S1", i32 1, i64 128, i64 64, i32 0, i32 0, i32 0, metadata !13, i32 0, i32 0} ; [ DW_TAG_structure_type ]
!13 = metadata !{metadata !14, metadata !17}
!14 = metadata !{i32 786445, metadata !42, null, metadata !1, metadata !"m", i32 2, i64 64, i64 64, i64 0, i32 0, metadata !15} ; [ DW_TAG_member ]
!15 = metadata !{i32 786447, null, null, metadata !2, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !16} ; [ DW_TAG_pointer_type ]
!16 = metadata !{i32 786468, null, null, metadata !2, metadata !"float", i32 0, i64 32, i64 32, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ]
!17 = metadata !{i32 786445, metadata !42, null, metadata !1, metadata !"nums", i32 3, i64 32, i64 32, i64 64, i32 0, metadata !5} ; [ DW_TAG_member ]
!18 = metadata !{i32 786689, metadata !0, metadata !"nums", metadata !1, i32 7, metadata !5, i32 0, metadata !32} ; [ DW_TAG_arg_variable ]
!19 = metadata !{i32 786484, i32 0, metadata !2, metadata !"p", metadata !"p", metadata !"", metadata !1, i32 14, metadata !11, i32 0, i32 1, %struct.S1* @p, null} ; [ DW_TAG_variable ]
!20 = metadata !{i32 7, i32 13, metadata !0, null}
!21 = metadata !{i32 7, i32 21, metadata !0, null}
!22 = metadata !{i32 9, i32 3, metadata !23, null}
!23 = metadata !{i32 786443, metadata !1, null, metadata !0, i32 8, i32 1, i32 0} ; [ DW_TAG_lexical_block ]
!24 = metadata !{metadata !"int", metadata !25}
!25 = metadata !{metadata !"omnipotent char", metadata !26}
!26 = metadata !{metadata !"Simple C/C++ TBAA", null}
!27 = metadata !{i32 10, i32 3, metadata !23, null}
!28 = metadata !{metadata !"any pointer", metadata !25}
!29 = metadata !{i32 11, i32 3, metadata !23, null}
!30 = metadata !{%struct.S1* @p}
!31 = metadata !{i32 7, i32 13, metadata !0, metadata !32}
!32 = metadata !{i32 16, i32 3, metadata !33, null}
!33 = metadata !{i32 786443, metadata !1, null, metadata !6, i32 15, i32 15, i32 1} ; [ DW_TAG_lexical_block ]
!34 = metadata !{i32 1}
!35 = metadata !{i32 7, i32 21, metadata !0, metadata !32}
!36 = metadata !{i32 9, i32 3, metadata !23, metadata !32}
!37 = metadata !{i32 10, i32 3, metadata !23, metadata !32}
!38 = metadata !{i32 17, i32 1, metadata !33, null}
!39 = metadata !{metadata !0, metadata !6}
!40 = metadata !{metadata !19}
!41 = metadata !{metadata !9, metadata !18}
!42 = metadata !{metadata !"nm2.c", metadata !"/private/tmp"}
