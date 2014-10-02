; RUN: llc < %s -mtriple=arm64-apple-darwin

; rdar://9146594

define void @drt_vsprintf() nounwind ssp {
entry:
  %do_tab_convert = alloca i32, align 4
  br i1 undef, label %if.then24, label %if.else295, !dbg !13

if.then24:                                        ; preds = %entry
  unreachable

if.else295:                                       ; preds = %entry
  call void @llvm.dbg.declare(metadata !{i32* %do_tab_convert}, metadata !16, metadata !{metadata !"0x102"}), !dbg !18
  store i32 0, i32* %do_tab_convert, align 4, !dbg !19
  unreachable
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.gv = !{!0}
!llvm.dbg.sp = !{!1, !7, !10, !11, !12}

!0 = metadata !{metadata !"0x34\00vsplive\00vsplive\00\00617\001\001", metadata !1, metadata !2, metadata !6, null, null} ; [ DW_TAG_variable ]
!1 = metadata !{metadata !"0x2e\00drt_vsprintf\00drt_vsprintf\00\00616\000\001\000\006\00256\000\000", metadata !20, metadata !2, metadata !4, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!2 = metadata !{metadata !"0x29", metadata !20} ; [ DW_TAG_file_type ]
!3 = metadata !{metadata !"0x11\0012\00clang version 3.0 (http://llvm.org/git/clang.git git:/git/puzzlebox/clang.git/ c4d1aea01c4444eb81bdbf391f1be309127c3cf1)\001\00\000\00\000", metadata !20, metadata !21, metadata !21, null, null, null} ; [ DW_TAG_compile_unit ]
!4 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !20, metadata !2, null, metadata !5, i32 0} ; [ DW_TAG_subroutine_type ]
!5 = metadata !{metadata !6}
!6 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, metadata !3} ; [ DW_TAG_base_type ]
!7 = metadata !{metadata !"0x2e\00putc_mem\00putc_mem\00\0030\001\001\000\006\00256\000\000", metadata !20, metadata !2, metadata !8, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!8 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !20, metadata !2, null, metadata !9, i32 0} ; [ DW_TAG_subroutine_type ]
!9 = metadata !{null}
!10 = metadata !{metadata !"0x2e\00print_double\00print_double\00\00203\001\001\000\006\00256\000\000", metadata !20, metadata !2, metadata !4, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!11 = metadata !{metadata !"0x2e\00print_number\00print_number\00\0075\001\001\000\006\00256\000\000", metadata !20, metadata !2, metadata !4, i32 0, null, null, null, null} ; [ DW_TAG_subprogram ]
!12 = metadata !{metadata !"0x2e\00get_flags\00get_flags\00\00508\001\001\000\006\00256\000\000", metadata !20, metadata !2, metadata !8, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!13 = metadata !{i32 653, i32 5, metadata !14, null}
!14 = metadata !{metadata !"0xb\00652\0035\002", metadata !20, metadata !15} ; [ DW_TAG_lexical_block ]
!15 = metadata !{metadata !"0xb\00616\001\000", metadata !20, metadata !1} ; [ DW_TAG_lexical_block ]
!16 = metadata !{metadata !"0x100\00do_tab_convert\00853\000", metadata !17, metadata !2, metadata !6} ; [ DW_TAG_auto_variable ]
!17 = metadata !{metadata !"0xb\00850\0012\0033", metadata !20, metadata !14} ; [ DW_TAG_lexical_block ]
!18 = metadata !{i32 853, i32 11, metadata !17, null}
!19 = metadata !{i32 853, i32 29, metadata !17, null}
!20 = metadata !{metadata !"print.i", metadata !"/Volumes/Ebi/echeng/radars/r9146594"}
!21 = metadata !{i32 0}
