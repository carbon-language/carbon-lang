; This test case checks debug info during register moves for an argument.
; RUN: %clang -arch x86_64 -mllvm -fast-isel=false  %s -c -o %t.o
; RUN: %clang -arch x86_64 %t.o -o %t.out
; RUN: %test_debuginfo %s %t.out
; XFAIL: *
; XTARGET: darwin
; Radar 8412415

target triple = "x86_64-apple-darwin10.0.0"

%struct._mtx = type { i64, i32, %struct.anon }
%struct.anon = type { i32, i32 }

define i32 @foobar(%struct._mtx* nocapture %mutex) nounwind readonly noinline ssp {
; DEBUGGER: break foobar
; DEBUGGER: r
; DEBUGGER: info address mutex
; CHECK:  Symbol "mutex" is 
; CHECK-NEXT: 
; CHECK-NEXT: in register
entry:
  tail call void @llvm.dbg.value(metadata !{%struct._mtx* %mutex}, i64 0, metadata !8), !dbg !29
  tail call void @llvm.dbg.value(metadata !30, i64 0, metadata !21), !dbg !31
  tail call void @llvm.dbg.value(metadata !32, i64 0, metadata !23), !dbg !33
  tail call void @llvm.dbg.value(metadata !32, i64 0, metadata !24), !dbg !34
  br label %do.body1, !dbg !37

do.body1:                                         ; preds = %entry, %do.body1
  %0 = phi i32 [ 0, %entry ], [ %inc, %do.body1 ]
  %r.1 = phi i32 [ 1, %entry ], [ %r.0, %do.body1 ]
  %inc = add i32 %0, 1
  %tmp2 = getelementptr inbounds %struct._mtx* %mutex, i64 0, i32 1, !dbg !35
  %tmp3 = load i32* %tmp2, align 4, !dbg !35
  %tobool = icmp eq i32 %tmp3, 0, !dbg !35
  %r.0 = select i1 %tobool, i32 %r.1, i32 2
  %call = tail call i32 @bar(i32 %r.0, i32 %0), !dbg !38
  %cmp = icmp slt i32 %inc, %call, !dbg !39
  br i1 %cmp, label %do.body1, label %do.end9, !dbg !39

do.end9:                                          ; preds = %do.body1
  tail call void @llvm.dbg.value(metadata !40, i64 0, metadata !21), !dbg !41
  tail call void @llvm.dbg.value(metadata !{i32 %call}, i64 0, metadata !24), !dbg !38
  tail call void @llvm.dbg.value(metadata !{i32 %inc}, i64 0, metadata !23), !dbg !42
  %add = add nsw i32 %r.0, %call, !dbg !43
  ret i32 %add, !dbg !43
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

define i32 @bar(i32 %i, i32 %j) nounwind readnone noinline ssp {
entry:
  tail call void @llvm.dbg.value(metadata !{i32 %i}, i64 0, metadata !25), !dbg !44
  tail call void @llvm.dbg.value(metadata !{i32 %j}, i64 0, metadata !26), !dbg !45
  %add = add nsw i32 %j, %i, !dbg !46
  ret i32 %add, !dbg !46
}

define i32 @main() nounwind readonly ssp {
entry:
  %m = alloca %struct._mtx, align 8
  call void @llvm.dbg.declare(metadata !{%struct._mtx* %m}, metadata !27), !dbg !48
  %tmp = getelementptr inbounds %struct._mtx* %m, i64 0, i32 1, !dbg !49
  store i32 0, i32* %tmp, align 8, !dbg !49
  %call = call i32 @foobar(%struct._mtx* %m), !dbg !50
  ret i32 %call, !dbg !50
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.sp = !{!0, !6, !7}
!llvm.dbg.lv.foobar = !{!8, !21, !23, !24}
!llvm.dbg.lv.bar = !{!25, !26}
!llvm.dbg.lv.main = !{!27}

!0 = metadata !{i32 524334, i32 0, metadata !1, metadata !"foobar", metadata !"foobar", metadata !"foobar", metadata !1, i32 12, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true, i32 (%struct._mtx*)* @foobar} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 524329, metadata !"mu.c", metadata !"/private/tmp", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 524305, i32 0, i32 12, metadata !"mu.c", metadata !"/private/tmp", metadata !"clang version 2.9 (trunk 114183)", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 524309, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 524324, metadata !1, metadata !"int", metadata !1, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 524334, i32 0, metadata !1, metadata !"bar", metadata !"bar", metadata !"bar", metadata !1, i32 26, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true, i32 (i32, i32)* @bar} ; [ DW_TAG_subprogram ]
!7 = metadata !{i32 524334, i32 0, metadata !1, metadata !"main", metadata !"main", metadata !"main", metadata !1, i32 30, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true, i32 ()* @main} ; [ DW_TAG_subprogram ]
!8 = metadata !{i32 524545, metadata !0, metadata !"mutex", metadata !1, i32 12, metadata !9} ; [ DW_TAG_arg_variable ]
!9 = metadata !{i32 524303, metadata !1, metadata !"", metadata !1, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !10} ; [ DW_TAG_pointer_type ]
!10 = metadata !{i32 524310, metadata !1, metadata !"mtx_t", metadata !1, i32 9, i64 0, i64 0, i64 0, i32 0, metadata !11} ; [ DW_TAG_typedef ]
!11 = metadata !{i32 524307, metadata !1, metadata !"_mtx", metadata !1, i32 2, i64 192, i64 64, i64 0, i32 0, null, metadata !12, i32 0, null} ; [ DW_TAG_structure_type ]
!12 = metadata !{metadata !13, metadata !15, metadata !16}
!13 = metadata !{i32 524301, metadata !1, metadata !"ptr", metadata !1, i32 3, i64 64, i64 64, i64 0, i32 0, metadata !14} ; [ DW_TAG_member ]
!14 = metadata !{i32 524324, metadata !1, metadata !"long unsigned int", metadata !1, i32 0, i64 64, i64 64, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!15 = metadata !{i32 524301, metadata !1, metadata !"waiters", metadata !1, i32 4, i64 32, i64 32, i64 64, i32 0, metadata !5} ; [ DW_TAG_member ]
!16 = metadata !{i32 524301, metadata !1, metadata !"mtxi", metadata !1, i32 8, i64 64, i64 32, i64 96, i32 0, metadata !17} ; [ DW_TAG_member ]
!17 = metadata !{i32 524307, metadata !11, metadata !"", metadata !1, i32 5, i64 64, i64 32, i64 0, i32 0, null, metadata !18, i32 0, null} ; [ DW_TAG_structure_type ]
!18 = metadata !{metadata !19, metadata !20}
!19 = metadata !{i32 524301, metadata !1, metadata !"tag", metadata !1, i32 6, i64 32, i64 32, i64 0, i32 0, metadata !5} ; [ DW_TAG_member ]
!20 = metadata !{i32 524301, metadata !1, metadata !"pad", metadata !1, i32 7, i64 32, i64 32, i64 32, i32 0, metadata !5} ; [ DW_TAG_member ]
!21 = metadata !{i32 524544, metadata !22, metadata !"r", metadata !1, i32 13, metadata !5} ; [ DW_TAG_auto_variable ]
!22 = metadata !{i32 524299, metadata !0, i32 12, i32 52, metadata !1, i32 0} ; [ DW_TAG_lexical_block ]
!23 = metadata !{i32 524544, metadata !22, metadata !"l", metadata !1, i32 14, metadata !5} ; [ DW_TAG_auto_variable ]
!24 = metadata !{i32 524544, metadata !22, metadata !"j", metadata !1, i32 15, metadata !5} ; [ DW_TAG_auto_variable ]
!25 = metadata !{i32 524545, metadata !6, metadata !"i", metadata !1, i32 26, metadata !5} ; [ DW_TAG_arg_variable ]
!26 = metadata !{i32 524545, metadata !6, metadata !"j", metadata !1, i32 26, metadata !5} ; [ DW_TAG_arg_variable ]
!27 = metadata !{i32 524544, metadata !28, metadata !"m", metadata !1, i32 31, metadata !10} ; [ DW_TAG_auto_variable ]
!28 = metadata !{i32 524299, metadata !7, i32 30, i32 12, metadata !1, i32 4} ; [ DW_TAG_lexical_block ]
!29 = metadata !{i32 12, i32 45, metadata !0, null}
!30 = metadata !{i32 1}
!31 = metadata !{i32 13, i32 12, metadata !22, null}
!32 = metadata !{i32 0}
!33 = metadata !{i32 14, i32 12, metadata !22, null}
!34 = metadata !{i32 15, i32 12, metadata !22, null}
!35 = metadata !{i32 18, i32 5, metadata !36, null}
!36 = metadata !{i32 524299, metadata !22, i32 17, i32 6, metadata !1, i32 2} ; [ DW_TAG_lexical_block ]
!37 = metadata !{i32 16, i32 3, metadata !22, null}
!38 = metadata !{i32 20, i32 5, metadata !36, null}
!39 = metadata !{i32 22, i32 3, metadata !36, null}
!40 = metadata !{i32 2}
!41 = metadata !{i32 19, i32 7, metadata !36, null}
!42 = metadata !{i32 21, i32 5, metadata !36, null}
!43 = metadata !{i32 23, i32 3, metadata !22, null}
!44 = metadata !{i32 26, i32 39, metadata !6, null}
!45 = metadata !{i32 26, i32 46, metadata !6, null}
!46 = metadata !{i32 27, i32 3, metadata !47, null}
!47 = metadata !{i32 524299, metadata !6, i32 26, i32 49, metadata !1, i32 3} ; [ DW_TAG_lexical_block ]
!48 = metadata !{i32 31, i32 9, metadata !28, null}
!49 = metadata !{i32 32, i32 3, metadata !28, null}
!50 = metadata !{i32 33, i32 3, metadata !28, null}

