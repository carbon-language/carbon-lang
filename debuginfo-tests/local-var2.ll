; This test case checks handling of llvm.dbg.declare intrinsic during isel.
; RUN: %clang -arch x86_64 -O0 -mllvm -fast-isel=false -g %s -c -o %t.o
; RUN: %clang -arch x86_64 %t.o -o %t.out
; RUN: %test_debuginfo %s %t.out
; XFAIL: *
; XTARGET: darwin

target triple = "x86_64-apple-darwin10.0.0"

define i32 @f1() nounwind ssp {
; DEBUGGER: break f1
; DEBUGGER: r
; DEBUGGER: n
; DEBUGGER: p i
; CHECK: $1 = 42
entry:
  %i = alloca i32, align 4
  call void @llvm.dbg.declare(metadata !{i32* %i}, metadata !10), !dbg !12
  store i32 42, i32* %i, align 4, !dbg !13
  %tmp = load i32* %i, align 4, !dbg !14
  ret i32 %tmp, !dbg !14
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

define i32 @f2() nounwind ssp {
; DEBUGGER: break f2
; DEBUGGER: c
; DEBUGGER: n
; DEBUGGER: p i
; CHECK: $2 = 42
entry:
  call void @llvm.dbg.declare(metadata !{i32* %i}, metadata !15), !dbg !17
  %i = alloca i32, align 4
  store i32 42, i32* %i, align 4, !dbg !18
  %tmp = load i32* %i, align 4, !dbg !19
  ret i32 %tmp, !dbg !19
}

; dbg.declare is dropped, as expected, by instruction selector.
; THIS IS NOT EXPECTED TO WORK.
define i32 @f3() nounwind ssp {
entry:
  call void @llvm.dbg.declare(metadata !{i32* %i}, metadata !20), !dbg !22
  br label %bbr
bbr:
  %i = alloca i32, align 4
  store i32 42, i32* %i, align 4, !dbg !23
  %tmp = load i32* %i, align 4, !dbg !24
  ret i32 %tmp, !dbg !24
}

; dbg.declare is dropped, as expected, by instruction selector.
; THIS IS NOT EXPECTED TO WORK.
define i32 @f4() nounwind ssp {
entry:
  %i = alloca i32, align 4
  call void @llvm.dbg.declare(metadata !{i32* %i}, metadata !25), !dbg !27
  ret i32 42, !dbg !28
}

define i32 @main() nounwind ssp {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  %call = call i32 @f1(), !dbg !29
  %call1 = call i32 @f2(), !dbg !31
  %call2 = call i32 @f3(), !dbg !32
  %call3 = call i32 @f4(), !dbg !33
  ret i32 0, !dbg !34
}

!llvm.dbg.sp = !{!0, !6, !7, !8, !9}

!0 = metadata !{i32 524334, i32 0, metadata !1, metadata !"f1", metadata !"f1", metadata !"f1", metadata !1, i32 2, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 ()* @f1} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 524329, metadata !"lv.c", metadata !"dbg_info_bugs", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 524305, i32 0, i32 12, metadata !"lv.c", metadata !"dbg_info_bugs", metadata !"clang version 2.9 (trunk 113428)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 524309, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 524324, metadata !1, metadata !"int", metadata !1, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 524334, i32 0, metadata !1, metadata !"f2", metadata !"f2", metadata !"f2", metadata !1, i32 8, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 ()* @f2} ; [ DW_TAG_subprogram ]
!7 = metadata !{i32 524334, i32 0, metadata !1, metadata !"f3", metadata !"f3", metadata !"f3", metadata !1, i32 14, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 ()* @f3} ; [ DW_TAG_subprogram ]
!8 = metadata !{i32 524334, i32 0, metadata !1, metadata !"f4", metadata !"f4", metadata !"f4", metadata !1, i32 20, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 ()* @f4} ; [ DW_TAG_subprogram ]
!9 = metadata !{i32 524334, i32 0, metadata !1, metadata !"main", metadata !"main", metadata !"main", metadata !1, i32 25, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 ()* @main} ; [ DW_TAG_subprogram ]
!10 = metadata !{i32 524544, metadata !11, metadata !"i", metadata !1, i32 3, metadata !5} ; [ DW_TAG_auto_variable ]
!11 = metadata !{i32 524299, metadata !0, i32 2, i32 10, metadata !1, i32 0} ; [ DW_TAG_lexical_block ]
!12 = metadata !{i32 3, i32 7, metadata !11, null}
!13 = metadata !{i32 4, i32 3, metadata !11, null}
!14 = metadata !{i32 5, i32 3, metadata !11, null}
!15 = metadata !{i32 524544, metadata !16, metadata !"i", metadata !1, i32 9, metadata !5} ; [ DW_TAG_auto_variable ]
!16 = metadata !{i32 524299, metadata !6, i32 8, i32 10, metadata !1, i32 1} ; [ DW_TAG_lexical_block ]
!17 = metadata !{i32 9, i32 7, metadata !16, null}
!18 = metadata !{i32 10, i32 3, metadata !16, null}
!19 = metadata !{i32 11, i32 3, metadata !16, null}
!20 = metadata !{i32 524544, metadata !21, metadata !"i", metadata !1, i32 15, metadata !5} ; [ DW_TAG_auto_variable ]
!21 = metadata !{i32 524299, metadata !7, i32 14, i32 10, metadata !1, i32 2} ; [ DW_TAG_lexical_block ]
!22 = metadata !{i32 15, i32 7, metadata !21, null}
!23 = metadata !{i32 16, i32 3, metadata !21, null}
!24 = metadata !{i32 17, i32 3, metadata !21, null}
!25 = metadata !{i32 524544, metadata !26, metadata !"i", metadata !1, i32 21, metadata !5} ; [ DW_TAG_auto_variable ]
!26 = metadata !{i32 524299, metadata !8, i32 20, i32 10, metadata !1, i32 3} ; [ DW_TAG_lexical_block ]
!27 = metadata !{i32 21, i32 7, metadata !26, null}
!28 = metadata !{i32 22, i32 3, metadata !26, null}
!29 = metadata !{i32 26, i32 3, metadata !30, null}
!30 = metadata !{i32 524299, metadata !9, i32 25, i32 12, metadata !1, i32 4} ; [ DW_TAG_lexical_block ]
!31 = metadata !{i32 27, i32 3, metadata !30, null}
!32 = metadata !{i32 28, i32 3, metadata !30, null}
!33 = metadata !{i32 29, i32 3, metadata !30, null}
!34 = metadata !{i32 30, i32 3, metadata !30, null}
