; RUN: llc < %s -break-anti-dependencies=all -march=ppc64 -mcpu=g5 | FileCheck %s
; CHECK-LABEL: main:

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind readnone {
entry:
  tail call void @llvm.dbg.value(metadata !{i32 %argc}, i64 0, metadata !15, metadata !{metadata !"0x102"}), !dbg !17
  tail call void @llvm.dbg.value(metadata !{i8** %argv}, i64 0, metadata !16, metadata !{metadata !"0x102"}), !dbg !18
  %add = add nsw i32 %argc, 1, !dbg !19
  ret i32 %add, !dbg !19
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.1\001\00\000\00\000", metadata !21, metadata !1, metadata !1, metadata !3, metadata !1, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x2e\00main\00main\00\001\000\001\000\006\00256\001\000", metadata !21, null, metadata !7, null, i32 (i32, i8**)* @main, null, null, metadata !13} ; [ DW_TAG_subprogram ]
!6 = metadata !{metadata !"0x29", metadata !21} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !9, metadata !10}
!9 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!10 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !11} ; [ DW_TAG_pointer_type ]
!11 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !12} ; [ DW_TAG_pointer_type ]
!12 = metadata !{metadata !"0x24\00char\000\008\008\000\000\008", null, null} ; [ DW_TAG_base_type ]
!13 = metadata !{metadata !15, metadata !16}
!15 = metadata !{metadata !"0x101\00argc\0016777217\000", metadata !5, metadata !6, metadata !9} ; [ DW_TAG_arg_variable ]
!16 = metadata !{metadata !"0x101\00argv\0033554433\000", metadata !5, metadata !6, metadata !10} ; [ DW_TAG_arg_variable ]
!17 = metadata !{i32 1, i32 14, metadata !5, null}
!18 = metadata !{i32 1, i32 26, metadata !5, null}
!19 = metadata !{i32 2, i32 3, metadata !20, null}
!20 = metadata !{metadata !"0xb\001\0034\000", metadata !21, metadata !5} ; [ DW_TAG_lexical_block ]
!21 = metadata !{metadata !"dbg.c", metadata !"/src"}
!22 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
