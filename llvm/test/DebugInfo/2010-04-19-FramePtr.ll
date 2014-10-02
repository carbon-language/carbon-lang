; RUN: %llc_dwarf -asm-verbose -O1 -o %t < %s
; RUN: grep DW_AT_APPLE_omit_frame_ptr %t
; RUN: %llc_dwarf -disable-fp-elim -asm-verbose -O1 -o %t < %s
; RUN: grep -v DW_AT_APPLE_omit_frame_ptr %t


define i32 @foo() nounwind ssp {
entry:
  %retval = alloca i32                            ; <i32*> [#uses=2]
  %0 = alloca i32                                 ; <i32*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store i32 42, i32* %0, align 4, !dbg !0
  %1 = load i32* %0, align 4, !dbg !0             ; <i32> [#uses=1]
  store i32 %1, i32* %retval, align 4, !dbg !0
  br label %return, !dbg !0

return:                                           ; preds = %entry
  %retval1 = load i32* %retval, !dbg !0           ; <i32> [#uses=1]
  ret i32 %retval1, !dbg !7
}

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!12}
!9 = metadata !{metadata !1}

!0 = metadata !{i32 2, i32 0, metadata !1, null}
!1 = metadata !{metadata !"0x2e\00foo\00foo\00foo\002\000\001\000\006\000\000\002", metadata !10, null, metadata !4, null, i32 ()* @foo, null, null, null} ; [ DW_TAG_subprogram ]
!2 = metadata !{metadata !"0x29", metadata !10} ; [ DW_TAG_file_type ]
!3 = metadata !{metadata !"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\000\00\000\00\000", metadata !10, metadata !11, metadata !11, metadata !9, null,  null} ; [ DW_TAG_compile_unit ]
!4 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !10, metadata !2, null, metadata !5, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = metadata !{metadata !6}
!6 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !10, metadata !2} ; [ DW_TAG_base_type ]
!7 = metadata !{i32 2, i32 0, metadata !8, null}
!8 = metadata !{metadata !"0xb\002\000\000", metadata !10, metadata !1} ; [ DW_TAG_lexical_block ]
!10 = metadata !{metadata !"a.c", metadata !"/tmp"}
!11 = metadata !{i32 0}
!12 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
