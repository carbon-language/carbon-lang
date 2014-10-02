; RUN: opt -licm -basicaa < %s -S | FileCheck %s

define void @dgefa() nounwind ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %for.cond.backedge, %entry
  br i1 undef, label %if.then, label %for.cond.backedge, !dbg !11

for.cond.backedge:                                ; preds = %for.body61, %for.body61.us, %for.body
  br i1 undef, label %for.end104, label %for.body, !dbg !15

if.then:                                          ; preds = %for.body
  br i1 undef, label %if.then27, label %if.end.if.end.split_crit_edge.critedge, !dbg !16

if.then27:                                        ; preds = %if.then
; CHECK: tail call void @llvm.dbg.value
  tail call void @llvm.dbg.value(metadata !18, i64 0, metadata !19, metadata !{}), !dbg !21
  br label %for.body61.us

if.end.if.end.split_crit_edge.critedge:           ; preds = %if.then
  br label %for.body61

for.body61.us:                                    ; preds = %for.body61.us, %if.then27
  br i1 undef, label %for.cond.backedge, label %for.body61.us, !dbg !23

for.body61:                                       ; preds = %for.body61, %if.end.if.end.split_crit_edge.critedge
  br i1 undef, label %for.cond.backedge, label %for.body61, !dbg !23

for.end104:                                       ; preds = %for.cond.backedge
  ret void, !dbg !24
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.module.flags = !{!26}
!llvm.dbg.sp = !{!0, !6, !9, !10}

!0 = metadata !{metadata !"0x2e\00idamax\00idamax\00\00112\000\001\000\006\00256\000\000", metadata !25, metadata !1, metadata !3, i32 0, null, null, null, null} ; [ DW_TAG_subprogram ]
!1 = metadata !{metadata !"0x29", metadata !25} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\0012\00clang version 2.9 (trunk 127169)\001\00\000\00\000", metadata !25, metadata !8, metadata !8, metadata !8, null, null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !25, metadata !1, null, metadata !4, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, metadata !2} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !"0x2e\00dscal\00dscal\00\00206\000\001\000\006\00256\000\000", metadata !25, metadata !1, metadata !7, i32 0, null, null, null, null} ; [ DW_TAG_subprogram ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !25, metadata !1, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null}
!9 = metadata !{metadata !"0x2e\00daxpy\00daxpy\00\00230\000\001\000\006\00256\000\000", metadata !25, metadata !1, metadata !7, i32 0, null, null, null, null} ; [ DW_TAG_subprogram ]
!10 = metadata !{metadata !"0x2e\00dgefa\00dgefa\00\00267\000\001\000\006\00256\000\000", metadata !25, metadata !1, metadata !7, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 267] [def] [scope 0] [dgefa]
!11 = metadata !{i32 281, i32 9, metadata !12, null}
!12 = metadata !{metadata !"0xb\00272\005\0032", metadata !25, metadata !13} ; [ DW_TAG_lexical_block ]
!13 = metadata !{metadata !"0xb\00271\005\0031", metadata !25, metadata !14} ; [ DW_TAG_lexical_block ]
!14 = metadata !{metadata !"0xb\00267\001\0030", metadata !25, metadata !10} ; [ DW_TAG_lexical_block ]
!15 = metadata !{i32 271, i32 5, metadata !14, null}
!16 = metadata !{i32 284, i32 10, metadata !17, null}
!17 = metadata !{metadata !"0xb\00282\009\0033", metadata !25, metadata !12} ; [ DW_TAG_lexical_block ]
!18 = metadata !{double undef}
!19 = metadata !{metadata !"0x100\00temp\00268\000", metadata !14, metadata !1, metadata !20} ; [ DW_TAG_auto_variable ]
!20 = metadata !{metadata !"0x24\00double\000\0064\0064\000\000\004", null, metadata !2} ; [ DW_TAG_base_type ]
!21 = metadata !{i32 286, i32 14, metadata !22, null}
!22 = metadata !{metadata !"0xb\00285\0013\0034", metadata !25, metadata !17} ; [ DW_TAG_lexical_block ]
!23 = metadata !{i32 296, i32 13, metadata !17, null}
!24 = metadata !{i32 313, i32 1, metadata !14, null}
!25 = metadata !{metadata !"/Volumes/Lalgate/work/llvm/projects/llvm-test/SingleSource/Benchmarks/CoyoteBench/lpbench.c", metadata !"/private/tmp"}
!26 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
