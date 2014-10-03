; RUN: llc < %s | FileCheck %s
; Should sink matching DBG_VALUEs also.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

define i32 @foo(i32 %i, i32* nocapture %c) nounwind uwtable readonly ssp {
  tail call void @llvm.dbg.value(metadata !{i32 %i}, i64 0, metadata !6, metadata !{metadata !"0x102"}), !dbg !12
  %ab = load i32* %c, align 1, !dbg !14
  tail call void @llvm.dbg.value(metadata !{i32* %c}, i64 0, metadata !7, metadata !{metadata !"0x102"}), !dbg !13
  tail call void @llvm.dbg.value(metadata !{i32 %ab}, i64 0, metadata !10, metadata !{metadata !"0x102"}), !dbg !14
  %cd = icmp eq i32 %i, 42, !dbg !15
  br i1 %cd, label %bb1, label %bb2, !dbg !15

bb1:                                     ; preds = %0
;CHECK: DEBUG_VALUE: a
;CHECK:      .loc	1 5 5
;CHECK-NEXT: addl
  %gh = add nsw i32 %ab, 2, !dbg !16
  br label %bb2, !dbg !16

bb2:
  %.0 = phi i32 [ %gh, %bb1 ], [ 0, %0 ]
  ret i32 %.0, !dbg !17
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22}

!0 = metadata !{metadata !"0x11\0012\00Apple clang version 3.0 (tags/Apple/clang-211.10.1) (based on LLVM 3.0svn)\001\00\000\00\001", metadata !20, metadata !21, metadata !21, metadata !18, null,  null} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !"0x2e\00foo\00foo\00\002\000\001\000\006\00256\001\000", metadata !20, metadata !2, metadata !3, null, i32 (i32, i32*)* @foo, null, null, metadata !19} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 0] [foo]
!2 = metadata !{metadata !"0x29", metadata !20} ; [ DW_TAG_file_type ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !20, metadata !2, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, metadata !0} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !"0x101\00i\0016777218\000", metadata !1, metadata !2, metadata !5} ; [ DW_TAG_arg_variable ]
!7 = metadata !{metadata !"0x101\00c\0033554434\000", metadata !1, metadata !2, metadata !8} ; [ DW_TAG_arg_variable ]
!8 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, metadata !0, metadata !9} ; [ DW_TAG_pointer_type ]
!9 = metadata !{metadata !"0x24\00char\000\008\008\000\000\006", null, metadata !0} ; [ DW_TAG_base_type ]
!10 = metadata !{metadata !"0x100\00a\003\000", metadata !11, metadata !2, metadata !9} ; [ DW_TAG_auto_variable ]
!11 = metadata !{metadata !"0xb\002\0025\000", metadata !20, metadata !1} ; [ DW_TAG_lexical_block ]
!12 = metadata !{i32 2, i32 13, metadata !1, null}
!13 = metadata !{i32 2, i32 22, metadata !1, null}
!14 = metadata !{i32 3, i32 14, metadata !11, null}
!15 = metadata !{i32 4, i32 3, metadata !11, null}
!16 = metadata !{i32 5, i32 5, metadata !11, null}
!17 = metadata !{i32 7, i32 1, metadata !11, null}
!18 = metadata !{metadata !1}
!19 = metadata !{metadata !6, metadata !7, metadata !10}
!20 = metadata !{metadata !"a.c", metadata !"/private/tmp"}
!21 = metadata !{i32 0}
!22 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
