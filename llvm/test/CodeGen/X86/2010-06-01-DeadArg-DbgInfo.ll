; RUN: llc -O2 < %s | FileCheck %s
; RUN: llc -O2 -regalloc=basic < %s | FileCheck %s
; Test to check that unused argument 'this' is not undefined in debug info.

target triple = "x86_64-apple-darwin10.2"
%struct.foo = type { i32 }

@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 (%struct.foo*, i32)* @_ZN3foo3bazEi to i8*)], section "llvm.metadata" ; <[1 x i8*]*> [#uses=0]

define i32 @_ZN3foo3bazEi(%struct.foo* nocapture %this, i32 %x) nounwind readnone optsize noinline ssp align 2 {
;CHECK: DEBUG_VALUE: baz:this <- RDI{{$}}
entry:
  tail call void @llvm.dbg.value(metadata !{%struct.foo* %this}, i64 0, metadata !15, metadata !{metadata !"0x102"})
  tail call void @llvm.dbg.value(metadata !{i32 %x}, i64 0, metadata !16, metadata !{metadata !"0x102"})
  %0 = mul nsw i32 %x, 7, !dbg !29                ; <i32> [#uses=1]
  %1 = add nsw i32 %0, 1, !dbg !29                ; <i32> [#uses=1]
  ret i32 %1, !dbg !29
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!34}
!llvm.dbg.lv = !{!0, !14, !15, !16, !17, !24, !25, !28}

!0 = metadata !{metadata !"0x101\00this\0011\000", metadata !1, metadata !3, metadata !12} ; [ DW_TAG_arg_variable ]
!1 = metadata !{metadata !"0x2e\00bar\00bar\00_ZN3foo3barEi\0011\000\001\000\006\000\001\0011", metadata !31, metadata !2, metadata !9, null, i32 (%struct.foo*, i32)* null, null, null, null} ; [ DW_TAG_subprogram ]
!2 = metadata !{metadata !"0x13\00foo\003\0032\0032\000\000\000", metadata !31, metadata !3, null, metadata !5, null, null, null} ; [ DW_TAG_structure_type ] [foo] [line 3, size 32, align 32, offset 0] [def] [from ]
!3 = metadata !{metadata !"0x29", metadata !31} ; [ DW_TAG_file_type ]
!4 = metadata !{metadata !"0x11\004\004.2.1 LLVM build\001\00\000\00\000", metadata !31, metadata !32, metadata !32, metadata !33, null, null} ; [ DW_TAG_compile_unit ]
!5 = metadata !{metadata !6, metadata !1, metadata !8}
!6 = metadata !{metadata !"0xd\00y\008\0032\0032\000\000", metadata !31, metadata !2, metadata !7} ; [ DW_TAG_member ]
!7 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !31, metadata !3} ; [ DW_TAG_base_type ]
!8 = metadata !{metadata !"0x2e\00baz\00baz\00_ZN3foo3bazEi\0015\000\001\000\006\000\001\0015", metadata !31, metadata !2, metadata !9, null, i32 (%struct.foo*, i32)* @_ZN3foo3bazEi, null, null, null} ; [ DW_TAG_subprogram ]
!9 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !31, metadata !3, null, metadata !10, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = metadata !{metadata !7, metadata !11, metadata !7}
!11 = metadata !{metadata !"0xf\00\000\0064\0064\000\0064", metadata !31, metadata !3, metadata !2} ; [ DW_TAG_pointer_type ]
!12 = metadata !{metadata !"0x26\00\000\0064\0064\000\0064", metadata !31, metadata !3, metadata !13} ; [ DW_TAG_const_type ]
!13 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", metadata !31, metadata !3, metadata !2} ; [ DW_TAG_pointer_type ]
!14 = metadata !{metadata !"0x101\00x\0011\000", metadata !1, metadata !3, metadata !7} ; [ DW_TAG_arg_variable ]
!15 = metadata !{metadata !"0x101\00this\0015\000", metadata !8, metadata !3, metadata !12} ; [ DW_TAG_arg_variable ]
!16 = metadata !{metadata !"0x101\00x\0015\000", metadata !8, metadata !3, metadata !7} ; [ DW_TAG_arg_variable ]
!17 = metadata !{metadata !"0x101\00argc\0019\000", metadata !18, metadata !3, metadata !7} ; [ DW_TAG_arg_variable ]
!18 = metadata !{metadata !"0x2e\00main\00main\00main\0019\000\001\000\006\000\001\0019", metadata !31, metadata !3, metadata !19, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!19 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !31, metadata !3, null, metadata !20, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!20 = metadata !{metadata !7, metadata !7, metadata !21}
!21 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", metadata !31, metadata !3, metadata !22} ; [ DW_TAG_pointer_type ]
!22 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", metadata !31, metadata !3, metadata !23} ; [ DW_TAG_pointer_type ]
!23 = metadata !{metadata !"0x24\00char\000\008\008\000\000\006", metadata !31, metadata !3} ; [ DW_TAG_base_type ]
!24 = metadata !{metadata !"0x101\00argv\0019\000", metadata !18, metadata !3, metadata !21} ; [ DW_TAG_arg_variable ]
!25 = metadata !{metadata !"0x100\00a\0020\000", metadata !26, metadata !3, metadata !2} ; [ DW_TAG_auto_variable ]
!26 = metadata !{metadata !"0xb\0019\000\000", metadata !31, metadata !27} ; [ DW_TAG_lexical_block ]
!27 = metadata !{metadata !"0xb\0019\000\000", metadata !31, metadata !18} ; [ DW_TAG_lexical_block ]
!28 = metadata !{metadata !"0x100\00b\0021\000", metadata !26, metadata !3, metadata !7} ; [ DW_TAG_auto_variable ]
!29 = metadata !{i32 16, i32 0, metadata !30, null}
!30 = metadata !{metadata !"0xb\0015\000\000", metadata !31, metadata !8} ; [ DW_TAG_lexical_block ]
!31 = metadata !{metadata !"foo.cp", metadata !"/tmp/"}
!32 = metadata !{i32 0}
!33 = metadata !{metadata !1, metadata !8, metadata !18}
!34 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
