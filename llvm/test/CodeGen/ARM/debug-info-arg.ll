; RUN: llc < %s | FileCheck %s
; Test to check argument y's debug info uses FI
; Radar 10048772
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-ios"

%struct.tag_s = type { i32, i32, i32 }

define void @foo(%struct.tag_s* nocapture %this, %struct.tag_s* %c, i64 %x, i64 %y, %struct.tag_s* nocapture %ptr1, %struct.tag_s* nocapture %ptr2) nounwind ssp {
  tail call void @llvm.dbg.value(metadata !{%struct.tag_s* %this}, i64 0, metadata !5, metadata !{metadata !"0x102"}), !dbg !20
  tail call void @llvm.dbg.value(metadata !{%struct.tag_s* %c}, i64 0, metadata !13, metadata !{metadata !"0x102"}), !dbg !21
  tail call void @llvm.dbg.value(metadata !{i64 %x}, i64 0, metadata !14, metadata !{metadata !"0x102"}), !dbg !22
  tail call void @llvm.dbg.value(metadata !{i64 %y}, i64 0, metadata !17, metadata !{metadata !"0x102"}), !dbg !23
;CHECK:	@DEBUG_VALUE: foo:y <- [R7+8]
  tail call void @llvm.dbg.value(metadata !{%struct.tag_s* %ptr1}, i64 0, metadata !18, metadata !{metadata !"0x102"}), !dbg !24
  tail call void @llvm.dbg.value(metadata !{%struct.tag_s* %ptr2}, i64 0, metadata !19, metadata !{metadata !"0x102"}), !dbg !25
  %1 = icmp eq %struct.tag_s* %c, null, !dbg !26
  br i1 %1, label %3, label %2, !dbg !26

; <label>:2                                       ; preds = %0
  tail call void @foobar(i64 %x, i64 %y) nounwind, !dbg !28
  br label %3, !dbg !28

; <label>:3                                       ; preds = %0, %2
  ret void, !dbg !29
}

declare void @foobar(i64, i64)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!33}

!0 = metadata !{metadata !"0x11\0012\00Apple clang version 3.0 (tags/Apple/clang-211.10.1) (based on LLVM 3.0svn)\001\00\000\00\001", metadata !32, metadata !4, metadata !4, metadata !30, null,  null} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !"0x2e\00foo\00foo\00\0011\000\001\000\006\00256\001\0011", metadata !2, metadata !2, metadata !3, null, void (%struct.tag_s*, %struct.tag_s*, i64, i64, %struct.tag_s*, %struct.tag_s*)* @foo, null, null, metadata !31} ; [ DW_TAG_subprogram ] [line 11] [def] [foo]
!2 = metadata !{metadata !"0x29", metadata !32} ; [ DW_TAG_file_type ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !32, metadata !2, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{null}
!5 = metadata !{metadata !"0x101\00this\0016777227\000", metadata !1, metadata !2, metadata !6} ; [ DW_TAG_arg_variable ]
!6 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", null, metadata !0, metadata !7} ; [ DW_TAG_pointer_type ]
!7 = metadata !{metadata !"0x13\00tag_s\005\0096\0032\000\000\000", metadata !32, metadata !0, null, metadata !8, null, null, null} ; [ DW_TAG_structure_type ] [tag_s] [line 5, size 96, align 32, offset 0] [def] [from ]
!8 = metadata !{metadata !9, metadata !11, metadata !12}
!9 = metadata !{metadata !"0xd\00x\006\0032\0032\000\000", metadata !32, metadata !7, metadata !10} ; [ DW_TAG_member ]
!10 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, metadata !0} ; [ DW_TAG_base_type ]
!11 = metadata !{metadata !"0xd\00y\007\0032\0032\0032\000", metadata !32, metadata !7, metadata !10} ; [ DW_TAG_member ]
!12 = metadata !{metadata !"0xd\00z\008\0032\0032\0064\000", metadata !32, metadata !7, metadata !10} ; [ DW_TAG_member ]
!13 = metadata !{metadata !"0x101\00c\0033554443\000", metadata !1, metadata !2, metadata !6} ; [ DW_TAG_arg_variable ]
!14 = metadata !{metadata !"0x101\00x\0050331659\000", metadata !1, metadata !2, metadata !15} ; [ DW_TAG_arg_variable ]
!15 = metadata !{metadata !"0x16\00UInt64\001\000\000\000\000", metadata !32, metadata !0, metadata !16} ; [ DW_TAG_typedef ]
!16 = metadata !{metadata !"0x24\00long long unsigned int\000\0064\0032\000\000\007", null, metadata !0} ; [ DW_TAG_base_type ]
!17 = metadata !{metadata !"0x101\00y\0067108875\000", metadata !1, metadata !2, metadata !15} ; [ DW_TAG_arg_variable ]
!18 = metadata !{metadata !"0x101\00ptr1\0083886091\000", metadata !1, metadata !2, metadata !6} ; [ DW_TAG_arg_variable ]
!19 = metadata !{metadata !"0x101\00ptr2\00100663307\000", metadata !1, metadata !2, metadata !6} ; [ DW_TAG_arg_variable ]
!20 = metadata !{i32 11, i32 24, metadata !1, null}
!21 = metadata !{i32 11, i32 44, metadata !1, null}
!22 = metadata !{i32 11, i32 54, metadata !1, null}
!23 = metadata !{i32 11, i32 64, metadata !1, null}
!24 = metadata !{i32 11, i32 81, metadata !1, null}
!25 = metadata !{i32 11, i32 101, metadata !1, null}
!26 = metadata !{i32 12, i32 3, metadata !27, null}
!27 = metadata !{metadata !"0xb\0011\00107\000", metadata !2, metadata !1} ; [ DW_TAG_lexical_block ]
!28 = metadata !{i32 13, i32 5, metadata !27, null}
!29 = metadata !{i32 14, i32 1, metadata !27, null}
!30 = metadata !{metadata !1}
!31 = metadata !{metadata !5, metadata !13, metadata !14, metadata !17, metadata !18, metadata!19}
!32 = metadata !{metadata !"one.c", metadata !"/Volumes/Athwagate/R10048772"}
!33 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
