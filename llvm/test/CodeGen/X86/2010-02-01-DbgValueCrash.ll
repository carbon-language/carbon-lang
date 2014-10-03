; RUN: llc -O1 < %s
; ModuleID = 'pr6157.bc'
; formerly crashed in SelectionDAGBuilder

%tart.reflect.ComplexType = type { double, double }

@.type.SwitchStmtTest = constant %tart.reflect.ComplexType { double 3.0, double 2.0 }

define i32 @"main(tart.core.String[])->int32"(i32 %args) {
entry:
  tail call void @llvm.dbg.value(metadata !14, i64 0, metadata !8, metadata !{metadata !"0x102"})
  tail call void @"tart.reflect.ComplexType.create->tart.core.Object"(%tart.reflect.ComplexType* @.type.SwitchStmtTest) ; <%tart.core.Object*> [#uses=2]
  ret i32 3
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone
declare void @"tart.reflect.ComplexType.create->tart.core.Object"(%tart.reflect.ComplexType*) nounwind readnone

!0 = metadata !{metadata !"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\001\00\000\00\000", metadata !15, metadata !16, metadata !16, null, null, null} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !"0x26\00\000\00192\0064\000\000", metadata !15, metadata !0, metadata !2} ; [ DW_TAG_const_type ]
!2 = metadata !{metadata !"0x13\00C\001\00192\0064\000\000\000", metadata !15, metadata !0, null, metadata !3, null, null, null} ; [ DW_TAG_structure_type ] [C] [line 1, size 192, align 64, offset 0] [def] [from ]
!3 = metadata !{metadata !4, metadata !6, metadata !7}
!4 = metadata !{metadata !"0xd\00x\001\0064\0064\000\000", metadata !15, metadata !2, metadata !5} ; [ DW_TAG_member ]
!5 = metadata !{metadata !"0x24\00double\000\0064\0064\000\000\004", metadata !15, metadata !0} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !"0xd\00y\001\0064\0064\0064\000", metadata !15, metadata !2, metadata !5} ; [ DW_TAG_member ]
!7 = metadata !{metadata !"0xd\00z\001\0064\0064\00128\000", metadata !15, metadata !2, metadata !5} ; [ DW_TAG_member ]
!8 = metadata !{metadata !"0x100\00t\005\000", metadata !9, metadata !0, metadata !2} ; [ DW_TAG_auto_variable ]
!9 = metadata !{metadata !"0xb\000\000\000", null, metadata !10}        ; [ DW_TAG_lexical_block ]
!10 = metadata !{metadata !"0x2e\00foo\00foo\00foo\004\000\001\000\006\000\000\000", i32 0, metadata !0, metadata !11, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!11 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !15, metadata !0, null, metadata !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{metadata !13}
!13 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !15, metadata !0} ; [ DW_TAG_base_type ]
!14 = metadata !{%tart.reflect.ComplexType* @.type.SwitchStmtTest}
!15 = metadata !{metadata !"sm.c", metadata !""}
!16 = metadata !{i32 0}
