; RUN: llc  -o /dev/null < %s
; Radar 7937664
%struct.AppleEvent = type opaque

define void @DisposeDMNotificationUPP(void (%struct.AppleEvent*)* %userUPP) "no-frame-pointer-elim-non-leaf" nounwind ssp {
entry:
  %userUPP_addr = alloca void (%struct.AppleEvent*)* ; <void (%struct.AppleEvent*)**> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata !{void (%struct.AppleEvent*)** %userUPP_addr}, metadata !0, metadata !{metadata !"0x102"}), !dbg !13
  store void (%struct.AppleEvent*)* %userUPP, void (%struct.AppleEvent*)** %userUPP_addr
  br label %return, !dbg !14

return:                                           ; preds = %entry
  ret void, !dbg !14
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!19}
!0 = metadata !{metadata !"0x101\00userUPP\007\000", metadata !1, metadata !2, metadata !6} ; [ DW_TAG_arg_variable ]
!1 = metadata !{metadata !"0x2e\00DisposeDMNotificationUPP\00DisposeDMNotificationUPP\00DisposeDMNotificationUPP\007\000\001\000\006\000\000\000", metadata !16, null, metadata !4, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!2 = metadata !{metadata !"0x29", metadata !16} ; [ DW_TAG_file_type ]
!3 = metadata !{metadata !"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build 9999)\001\00\000\00\000", metadata !16, metadata !17, metadata !17, metadata !18, null, null} ; [ DW_TAG_compile_unit ]
!4 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !16, metadata !2, null, metadata !5, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = metadata !{null, metadata !6}
!6 = metadata !{metadata !"0x16\00DMNotificationUPP\006\000\000\000\000", metadata !16, metadata !2, metadata !7} ; [ DW_TAG_typedef ]
!7 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", metadata !16, metadata !2, metadata !8} ; [ DW_TAG_pointer_type ]
!8 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !16, metadata !2, null, metadata !9, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!9 = metadata !{null, metadata !10}
!10 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", metadata !16, metadata !2, metadata !11} ; [ DW_TAG_pointer_type ]
!11 = metadata !{metadata !"0x16\00AppleEvent\004\000\000\000\000", metadata !16, metadata !2, metadata !12} ; [ DW_TAG_typedef ]
!12 = metadata !{metadata !"0x13\00AEDesc\001\000\000\000\004\000", metadata !16, metadata !2, null, null, null, null, null} ; [ DW_TAG_structure_type ] [AEDesc] [line 1, size 0, align 0, offset 0] [decl] [from ]
!13 = metadata !{i32 7, i32 0, metadata !1, null}
!14 = metadata !{i32 8, i32 0, metadata !15, null}
!15 = metadata !{metadata !"0xb\007\000\000", metadata !16, metadata !1} ; [ DW_TAG_lexical_block ]
!16 = metadata !{metadata !"t.c", metadata !"/Users/echeng/LLVM/radars/r7937664/"}
!17 = metadata !{i32 0}
!18 = metadata !{metadata !1}
!19 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
