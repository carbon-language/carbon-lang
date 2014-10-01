; RUN: llc < %s -mtriple=arm64-apple-darwin

; rdar://9146594

define void @drt_vsprintf() nounwind ssp {
entry:
  %do_tab_convert = alloca i32, align 4
  br i1 undef, label %if.then24, label %if.else295, !dbg !13

if.then24:                                        ; preds = %entry
  unreachable

if.else295:                                       ; preds = %entry
  call void @llvm.dbg.declare(metadata !{i32* %do_tab_convert}, metadata !16), !dbg !18
  store i32 0, i32* %do_tab_convert, align 4, !dbg !19
  unreachable
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!llvm.dbg.gv = !{!0}
!llvm.dbg.sp = !{!1, !7, !10, !11, !12}

!0 = metadata !{i32 589876, i32 0, metadata !1, metadata !"vsplive", metadata !"vsplive", metadata !"", metadata !2, i32 617, metadata !6, i32 1, i32 1, null, null} ; [ DW_TAG_variable ]
!1 = metadata !{i32 589870, metadata !20, metadata !2, metadata !"drt_vsprintf", metadata !"drt_vsprintf", metadata !"", i32 616, metadata !4, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 589865, metadata !20} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 589841, metadata !20, i32 12, metadata !"clang version 3.0 (http://llvm.org/git/clang.git git:/git/puzzlebox/clang.git/ c4d1aea01c4444eb81bdbf391f1be309127c3cf1)", i1 true, metadata !"", i32 0, metadata !21, metadata !21, null, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!4 = metadata !{i32 589845, metadata !20, metadata !2, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !5, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!5 = metadata !{metadata !6}
!6 = metadata !{i32 589860, null, metadata !3, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!7 = metadata !{i32 589870, metadata !20, metadata !2, metadata !"putc_mem", metadata !"putc_mem", metadata !"", i32 30, metadata !8, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!8 = metadata !{i32 589845, metadata !20, metadata !2, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !9, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!9 = metadata !{null}
!10 = metadata !{i32 589870, metadata !20, metadata !2, metadata !"print_double", metadata !"print_double", metadata !"", i32 203, metadata !4, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!11 = metadata !{i32 589870, metadata !20, metadata !2, metadata !"print_number", metadata !"print_number", metadata !"", i32 75, metadata !4, i1 true, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, null, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!12 = metadata !{i32 589870, metadata !20, metadata !2, metadata !"get_flags", metadata !"get_flags", metadata !"", i32 508, metadata !8, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!13 = metadata !{i32 653, i32 5, metadata !14, null}
!14 = metadata !{i32 589835, metadata !20, metadata !15, i32 652, i32 35, i32 2} ; [ DW_TAG_lexical_block ]
!15 = metadata !{i32 589835, metadata !20, metadata !1, i32 616, i32 1, i32 0} ; [ DW_TAG_lexical_block ]
!16 = metadata !{i32 590080, metadata !17, metadata !"do_tab_convert", metadata !2, i32 853, metadata !6, i32 0, null} ; [ DW_TAG_auto_variable ]
!17 = metadata !{i32 589835, metadata !20, metadata !14, i32 850, i32 12, i32 33} ; [ DW_TAG_lexical_block ]
!18 = metadata !{i32 853, i32 11, metadata !17, null}
!19 = metadata !{i32 853, i32 29, metadata !17, null}
!20 = metadata !{metadata !"print.i", metadata !"/Volumes/Ebi/echeng/radars/r9146594"}
!21 = metadata !{i32 0}
