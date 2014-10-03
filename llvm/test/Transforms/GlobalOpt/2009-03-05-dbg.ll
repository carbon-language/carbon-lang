; REQUIRES: asserts
; RUN: opt < %s -globalopt -stats -disable-output 2>&1 | grep "1 globalopt - Number of global vars shrunk to booleans"

@Stop = internal global i32 0                     ; <i32*> [#uses=3]

define i32 @foo(i32 %i) nounwind ssp {
entry:
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.value(metadata !{i32 %i}, i64 0, metadata !3, metadata !{})
  %0 = icmp eq i32 %i, 1, !dbg !7                 ; <i1> [#uses=1]
  br i1 %0, label %bb, label %bb1, !dbg !7

bb:                                               ; preds = %entry
  store i32 0, i32* @Stop, align 4, !dbg !9
  %1 = mul nsw i32 %i, 42, !dbg !10               ; <i32> [#uses=1]
  call void @llvm.dbg.value(metadata !{i32 %1}, i64 0, metadata !3, metadata !{}), !dbg !10
  br label %bb2, !dbg !10

bb1:                                              ; preds = %entry
  store i32 1, i32* @Stop, align 4, !dbg !11
  br label %bb2, !dbg !11

bb2:                                              ; preds = %bb1, %bb
  %i_addr.0 = phi i32 [ %1, %bb ], [ %i, %bb1 ]   ; <i32> [#uses=1]
  br label %return, !dbg !12

return:                                           ; preds = %bb2
  ret i32 %i_addr.0, !dbg !12
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define i32 @bar() nounwind ssp {
entry:
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %0 = load i32* @Stop, align 4, !dbg !13         ; <i32> [#uses=1]
  %1 = icmp eq i32 %0, 1, !dbg !13                ; <i1> [#uses=1]
  br i1 %1, label %bb, label %bb1, !dbg !13

bb:                                               ; preds = %entry
  br label %bb2, !dbg !18

bb1:                                              ; preds = %entry
  br label %bb2, !dbg !19

bb2:                                              ; preds = %bb1, %bb
  %.0 = phi i32 [ 0, %bb ], [ 1, %bb1 ]           ; <i32> [#uses=1]
  br label %return, !dbg !19

return:                                           ; preds = %bb2
  ret i32 %.0, !dbg !19
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.gv = !{!0}

!0 = metadata !{metadata !"0x34\00Stop\00Stop\00\002\001\001", metadata !1, metadata !1, metadata !2, i32* @Stop} ; [ DW_TAG_variable ]
!1 = metadata !{metadata !"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\001\00\000\00\000", metadata !20, metadata !21, metadata !21, null, null, null} ; [ DW_TAG_compile_unit ]
!2 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, metadata !1} ; [ DW_TAG_base_type ]
!3 = metadata !{metadata !"0x101\00i\004\000", metadata !4, metadata !1, metadata !2} ; [ DW_TAG_arg_variable ]
!4 = metadata !{metadata !"0x2e\00foo\00foo\00foo\004\000\001\000\006\000\000\000", i32 0, metadata !1, metadata !5, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!5 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !1, null, null, metadata !6, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!6 = metadata !{metadata !2, metadata !2}
!7 = metadata !{i32 5, i32 0, metadata !8, null}
!8 = metadata !{metadata !"0xb\000\000\000", metadata !20, metadata !4} ; [ DW_TAG_lexical_block ]
!9 = metadata !{i32 6, i32 0, metadata !8, null}
!10 = metadata !{i32 7, i32 0, metadata !8, null}
!11 = metadata !{i32 9, i32 0, metadata !8, null}
!12 = metadata !{i32 11, i32 0, metadata !8, null}
!13 = metadata !{i32 14, i32 0, metadata !14, null}
!14 = metadata !{metadata !"0xb\000\000\000", metadata !20, metadata !15} ; [ DW_TAG_lexical_block ]
!15 = metadata !{metadata !"0x2e\00bar\00bar\00bar\0013\000\001\000\006\000\000\000", i32 0, metadata !1, metadata !16, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!16 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !1, null, null, metadata !17, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!17 = metadata !{metadata !2}
!18 = metadata !{i32 15, i32 0, metadata !14, null}
!19 = metadata !{i32 16, i32 0, metadata !14, null}
!20 = metadata !{metadata !"g.c", metadata !"/tmp"}
!21 = metadata !{i32 0}
