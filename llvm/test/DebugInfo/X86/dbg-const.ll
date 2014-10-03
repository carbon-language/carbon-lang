; RUN: llc < %s - | FileCheck %s
;
; FIXME: A potentially more interesting test case would be:
; %call = @bar()
; dbg.value j=0
; %call2 = @bar()
; dbg.value j=%call
;
; We cannot current handle the above sequence because codegenprepare
; hoists the second dbg.value above %call2, which then appears to
; conflict with j=0. It does this because SelectionDAG cannot handle
; global debug values.

target triple = "x86_64-apple-darwin10.0.0"

;CHECK:        ## DW_OP_consts
;CHECK-NEXT:  .byte	42
define i32 @foobar() nounwind readonly noinline ssp {
entry:
  tail call void @llvm.dbg.value(metadata !8, i64 0, metadata !6, metadata !{metadata !"0x102"}), !dbg !9
  %call = tail call i32 @bar(), !dbg !11
  tail call void @llvm.dbg.value(metadata !{i32 %call}, i64 0, metadata !6, metadata !{metadata !"0x102"}), !dbg !11
  %call2 = tail call i32 @bar(), !dbg !11
  %add = add nsw i32 %call2, %call, !dbg !12
  ret i32 %add, !dbg !10
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone
declare i32 @bar() nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!17}

!0 = metadata !{metadata !"0x2e\00foobar\00foobar\00foobar\0012\000\001\000\006\000\001\000", metadata !15, metadata !1, metadata !3, null, i32 ()* @foobar, null, null, metadata !14} ; [ DW_TAG_subprogram ]
!1 = metadata !{metadata !"0x29", metadata !15} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\0012\00clang version 2.9 (trunk 114183)\001\00\000\00\001", metadata !15, metadata !16, metadata !16, metadata !13, null,  null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !15, metadata !1, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !15, metadata !1} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !"0x100\00j\0015\000", metadata !7, metadata !1, metadata !5} ; [ DW_TAG_auto_variable ]
!7 = metadata !{metadata !"0xb\0012\0052\000", metadata !15, metadata !0} ; [ DW_TAG_lexical_block ]
!8 = metadata !{i32 42}
!9 = metadata !{i32 15, i32 12, metadata !7, null}
!10 = metadata !{i32 23, i32 3, metadata !7, null}
!11 = metadata !{i32 17, i32 3, metadata !7, null}
!12 = metadata !{i32 18, i32 3, metadata !7, null}
!13 = metadata !{metadata !0}
!14 = metadata !{metadata !6}
!15 = metadata !{metadata !"mu.c", metadata !"/private/tmp"}
!16 = metadata !{i32 0}
!17 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
