; RUN: llc -O2 -mtriple=i386-apple-darwin <%s | FileCheck %s
; Use DW_FORM_addr for DW_AT_entry_pc.
; Radar 8094785

; CHECK:	.byte	17                      ## DW_TAG_compile_unit
; CHECK-NEXT:	.byte	1                       ## DW_CHILDREN_yes
; CHECK-NEXT:	.byte	37                      ## DW_AT_producer
; CHECK-NEXT:	.byte	8                       ## DW_FORM_string
; CHECK-NEXT:	.byte	19                      ## DW_AT_language
; CHECK-NEXT:	.byte	11                      ## DW_FORM_data1
; CHECK-NEXT:	.byte	3                       ## DW_AT_name
; CHECK-NEXT:	.byte	8                       ## DW_FORM_string
; CHECK-NEXT:	.byte	82                      ## DW_AT_entry_pc
; CHECK-NEXT:	.byte	1                       ## DW_FORM_addr
; CHECK-NEXT:	.byte	16                      ## DW_AT_stmt_list
; CHECK-NEXT:	.byte	6                       ## DW_FORM_data4
; CHECK-NEXT:	.byte	27                      ## DW_AT_comp_dir
; CHECK-NEXT:	.byte	8                       ## DW_FORM_string
; CHECK-NEXT:	.byte	225                     ## DW_AT_APPLE_optimized

%struct.a = type { i32, %struct.a* }

@ret = common global i32 0                        ; <i32*> [#uses=2]

define void @foo(i32 %x) nounwind noinline ssp {
entry:
  tail call void @llvm.dbg.value(metadata !{i32 %x}, i64 0, metadata !21), !dbg !28
  store i32 %x, i32* @ret, align 4, !dbg !29
  ret void, !dbg !31
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

define i8* @bar(%struct.a* %b) nounwind noinline ssp {
entry:
  tail call void @llvm.dbg.value(metadata !{%struct.a* %b}, i64 0, metadata !22), !dbg !32
  %0 = getelementptr inbounds %struct.a* %b, i64 0, i32 0, !dbg !33 ; <i32*> [#uses=1]
  %1 = load i32* %0, align 8, !dbg !33            ; <i32> [#uses=1]
  tail call void @foo(i32 %1) nounwind noinline ssp, !dbg !33
  %2 = bitcast %struct.a* %b to i8*, !dbg !35     ; <i8*> [#uses=1]
  ret i8* %2, !dbg !35
}

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind ssp {
entry:
  %e = alloca %struct.a, align 8                  ; <%struct.a*> [#uses=4]
  call void @llvm.dbg.value(metadata !{i32 %argc}, i64 0, metadata !23), !dbg !36
  call void @llvm.dbg.value(metadata !{i8** %argv}, i64 0, metadata !24), !dbg !36
  call void @llvm.dbg.declare(metadata !{%struct.a* %e}, metadata !25), !dbg !37
  %0 = getelementptr inbounds %struct.a* %e, i64 0, i32 0, !dbg !38 ; <i32*> [#uses=1]
  store i32 4, i32* %0, align 8, !dbg !38
  %1 = getelementptr inbounds %struct.a* %e, i64 0, i32 1, !dbg !39 ; <%struct.a**> [#uses=1]
  store %struct.a* %e, %struct.a** %1, align 8, !dbg !39
  %2 = call i8* @bar(%struct.a* %e) nounwind noinline ssp, !dbg !40 ; <i8*> [#uses=0]
  %3 = load i32* @ret, align 4, !dbg !41          ; <i32> [#uses=1]
  ret i32 %3, !dbg !41
}

!llvm.dbg.sp = !{!0, !6, !15}
!llvm.dbg.lv.foo = !{!21}
!llvm.dbg.lv.bar = !{!22}
!llvm.dbg.lv.main = !{!23, !24, !25}
!llvm.dbg.gv = !{!27}

!0 = metadata !{i32 524334, i32 0, metadata !1, metadata !"foo", metadata !"foo", metadata !"foo", metadata !1, i32 34, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true, void (i32)* @foo} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 524329, metadata !"2010-06-28-DbgEntryPC.c", metadata !"/Users/yash/clean/llvm/test/FrontendC", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 524305, i32 0, i32 1, metadata !"2010-06-28-DbgEntryPC.c", metadata !"/Users/yash/clean/llvm/test/FrontendC", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 524309, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{null, metadata !5}
!5 = metadata !{i32 524324, metadata !1, metadata !"int", metadata !1, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 524334, i32 0, metadata !1, metadata !"bar", metadata !"bar", metadata !"bar", metadata !1, i32 38, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true, i8* (%struct.a*)* @bar} ; [ DW_TAG_subprogram ]
!7 = metadata !{i32 524309, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{metadata !9, metadata !10}
!9 = metadata !{i32 524303, metadata !1, metadata !"", metadata !1, i32 0, i64 64, i64 64, i64 0, i32 0, null} ; [ DW_TAG_pointer_type ]
!10 = metadata !{i32 524303, metadata !1, metadata !"", metadata !1, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !11} ; [ DW_TAG_pointer_type ]
!11 = metadata !{i32 524307, metadata !1, metadata !"a", metadata !1, i32 23, i64 128, i64 64, i64 0, i32 0, null, metadata !12, i32 0, null} ; [ DW_TAG_structure_type ]
!12 = metadata !{metadata !13, metadata !14}
!13 = metadata !{i32 524301, metadata !11, metadata !"c", metadata !1, i32 24, i64 32, i64 32, i64 0, i32 0, metadata !5} ; [ DW_TAG_member ]
!14 = metadata !{i32 524301, metadata !11, metadata !"d", metadata !1, i32 25, i64 64, i64 64, i64 64, i32 0, metadata !10} ; [ DW_TAG_member ]
!15 = metadata !{i32 524334, i32 0, metadata !1, metadata !"main", metadata !"main", metadata !"main", metadata !1, i32 43, metadata !16, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true, i32 (i32, i8**)* @main} ; [ DW_TAG_subprogram ]
!16 = metadata !{i32 524309, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !17, i32 0, null} ; [ DW_TAG_subroutine_type ]
!17 = metadata !{metadata !5, metadata !5, metadata !18}
!18 = metadata !{i32 524303, metadata !1, metadata !"", metadata !1, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !19} ; [ DW_TAG_pointer_type ]
!19 = metadata !{i32 524303, metadata !1, metadata !"", metadata !1, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !20} ; [ DW_TAG_pointer_type ]
!20 = metadata !{i32 524324, metadata !1, metadata !"char", metadata !1, i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
!21 = metadata !{i32 524545, metadata !0, metadata !"x", metadata !1, i32 33, metadata !5} ; [ DW_TAG_arg_variable ]
!22 = metadata !{i32 524545, metadata !6, metadata !"b", metadata !1, i32 38, metadata !10} ; [ DW_TAG_arg_variable ]
!23 = metadata !{i32 524545, metadata !15, metadata !"argc", metadata !1, i32 43, metadata !5} ; [ DW_TAG_arg_variable ]
!24 = metadata !{i32 524545, metadata !15, metadata !"argv", metadata !1, i32 43, metadata !18} ; [ DW_TAG_arg_variable ]
!25 = metadata !{i32 524544, metadata !26, metadata !"e", metadata !1, i32 44, metadata !11} ; [ DW_TAG_auto_variable ]
!26 = metadata !{i32 524299, metadata !15, i32 43, i32 0} ; [ DW_TAG_lexical_block ]
!27 = metadata !{i32 524340, i32 0, metadata !1, metadata !"ret", metadata !"ret", metadata !"", metadata !1, i32 28, metadata !5, i1 false, i1 true, i32* @ret} ; [ DW_TAG_variable ]
!28 = metadata !{i32 33, i32 0, metadata !0, null}
!29 = metadata !{i32 35, i32 0, metadata !30, null}
!30 = metadata !{i32 524299, metadata !0, i32 34, i32 0} ; [ DW_TAG_lexical_block ]
!31 = metadata !{i32 36, i32 0, metadata !30, null}
!32 = metadata !{i32 38, i32 0, metadata !6, null}
!33 = metadata !{i32 39, i32 0, metadata !34, null}
!34 = metadata !{i32 524299, metadata !6, i32 38, i32 0} ; [ DW_TAG_lexical_block ]
!35 = metadata !{i32 40, i32 0, metadata !34, null}
!36 = metadata !{i32 43, i32 0, metadata !15, null}
!37 = metadata !{i32 44, i32 0, metadata !26, null}
!38 = metadata !{i32 45, i32 0, metadata !26, null}
!39 = metadata !{i32 46, i32 0, metadata !26, null}
!40 = metadata !{i32 48, i32 0, metadata !26, null}
!41 = metadata !{i32 49, i32 0, metadata !26, null}
