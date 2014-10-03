; RUN: llc -mtriple=x86_64-apple-darwin < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-apple-darwin -regalloc=basic < %s | FileCheck %s
; Test to check separate label for inlined function argument.

define i32 @foo(i32 %y) nounwind optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata !{i32 %y}, i64 0, metadata !0, metadata !{metadata !"0x102"})
  %0 = tail call i32 (...)* @zoo(i32 %y) nounwind, !dbg !9 ; <i32> [#uses=1]
  ret i32 %0, !dbg !9
}

declare i32 @zoo(...)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

define i32 @bar(i32 %x) nounwind optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata !{i32 %x}, i64 0, metadata !7, metadata !{metadata !"0x102"})
  tail call void @llvm.dbg.value(metadata !11, i64 0, metadata !0, metadata !{metadata !"0x102"}) nounwind
  %0 = tail call i32 (...)* @zoo(i32 1) nounwind, !dbg !12 ; <i32> [#uses=1]
  %1 = add nsw i32 %0, %x, !dbg !13               ; <i32> [#uses=1]
  ret i32 %1, !dbg !13
}

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!20}

!0 = metadata !{metadata !"0x101\00y\002\000", metadata !1, metadata !2, metadata !6} ; [ DW_TAG_arg_variable ]
!1 = metadata !{metadata !"0x2e\00foo\00foo\00foo\002\000\001\000\006\000\001\002", metadata !18, metadata !2, metadata !4, null, i32 (i32)* @foo, null, null, metadata !15} ; [ DW_TAG_subprogram ]
!2 = metadata !{metadata !"0x29", metadata !18} ; [ DW_TAG_file_type ]
!3 = metadata !{metadata !"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\001\00\000\00\001", metadata !18, metadata !19, metadata !19, metadata !17, null,  null} ; [ DW_TAG_compile_unit ]
!4 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !18, metadata !2, null, metadata !5, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = metadata !{metadata !6, metadata !6}
!6 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !18, metadata !2} ; [ DW_TAG_base_type ]
!7 = metadata !{metadata !"0x101\00x\006\000", metadata !8, metadata !2, metadata !6} ; [ DW_TAG_arg_variable ]
!8 = metadata !{metadata !"0x2e\00bar\00bar\00bar\006\000\001\000\006\000\001\006", metadata !18, metadata !2, metadata !4, null, i32 (i32)* @bar, null, null, metadata !16} ; [ DW_TAG_subprogram ]
!9 = metadata !{i32 3, i32 0, metadata !10, null}
!10 = metadata !{metadata !"0xb\002\000\000", metadata !18, metadata !1} ; [ DW_TAG_lexical_block ]
!11 = metadata !{i32 1}
!12 = metadata !{i32 3, i32 0, metadata !10, metadata !13}
!13 = metadata !{i32 7, i32 0, metadata !14, null}
!14 = metadata !{metadata !"0xb\006\000\000", metadata !18, metadata !8} ; [ DW_TAG_lexical_block ]
!15 = metadata !{metadata !0}
!16 = metadata !{metadata !7}
!17 = metadata !{metadata !1, metadata !8}
!18 = metadata !{metadata !"f.c", metadata !"/tmp"}
!19 = metadata !{i32 0}

;CHECK: DEBUG_VALUE: bar:x <- E
;CHECK: Ltmp
;CHECK:	DEBUG_VALUE: foo:y <- 1{{$}}
!20 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
