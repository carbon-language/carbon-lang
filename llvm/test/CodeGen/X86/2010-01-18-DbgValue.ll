; RUN: llc -march=x86 -O0 < %s | FileCheck %s
; Currently, dbg.declare generates a DEBUG_VALUE comment.  Eventually it will
; generate DWARF and this test will need to be modified or removed.


%struct.Pt = type { double, double }
%struct.Rect = type { %struct.Pt, %struct.Pt }

define double @foo(%struct.Rect* byval %my_r0) nounwind ssp {
entry:
;CHECK: DEBUG_VALUE
  %retval = alloca double                         ; <double*> [#uses=2]
  %0 = alloca double                              ; <double*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata !{%struct.Rect* %my_r0}, metadata !0, metadata !{metadata !"0x102"}), !dbg !15
  %1 = getelementptr inbounds %struct.Rect* %my_r0, i32 0, i32 0, !dbg !16 ; <%struct.Pt*> [#uses=1]
  %2 = getelementptr inbounds %struct.Pt* %1, i32 0, i32 0, !dbg !16 ; <double*> [#uses=1]
  %3 = load double* %2, align 8, !dbg !16         ; <double> [#uses=1]
  store double %3, double* %0, align 8, !dbg !16
  %4 = load double* %0, align 8, !dbg !16         ; <double> [#uses=1]
  store double %4, double* %retval, align 8, !dbg !16
  br label %return, !dbg !16

return:                                           ; preds = %entry
  %retval1 = load double* %retval, !dbg !16       ; <double> [#uses=1]
  ret double %retval1, !dbg !16
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!21}

!0 = metadata !{metadata !"0x101\00my_r0\0011\000", metadata !1, metadata !2, metadata !7} ; [ DW_TAG_arg_variable ]
!1 = metadata !{metadata !"0x2e\00foo\00foo\00foo\0011\000\001\000\006\000\000\0011", metadata !19, metadata !2, metadata !4, null, double (%struct.Rect*)* @foo, null, null, null} ; [ DW_TAG_subprogram ]
!2 = metadata !{metadata !"0x29", metadata !19} ; [ DW_TAG_file_type ]
!3 = metadata !{metadata !"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\000\00\000\00\000", metadata !19, metadata !20, metadata !20, metadata !18, null, null} ; [ DW_TAG_compile_unit ]
!4 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !19, metadata !2, null, metadata !5, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = metadata !{metadata !6, metadata !7}
!6 = metadata !{metadata !"0x24\00double\000\0064\0064\000\000\004", metadata !19, metadata !2} ; [ DW_TAG_base_type ]
!7 = metadata !{metadata !"0x13\00Rect\006\00256\0064\000\000\000", metadata !19, metadata !2, null, metadata !8, null, null, null} ; [ DW_TAG_structure_type ] [Rect] [line 6, size 256, align 64, offset 0] [def] [from ]
!8 = metadata !{metadata !9, metadata !14}
!9 = metadata !{metadata !"0xd\00P1\007\00128\0064\000\000", metadata !19, metadata !7, metadata !10} ; [ DW_TAG_member ]
!10 = metadata !{metadata !"0x13\00Pt\001\00128\0064\000\000\000", metadata !19, metadata !2, null, metadata !11, null, null, null} ; [ DW_TAG_structure_type ] [Pt] [line 1, size 128, align 64, offset 0] [def] [from ]
!11 = metadata !{metadata !12, metadata !13}
!12 = metadata !{metadata !"0xd\00x\002\0064\0064\000\000", metadata !19, metadata !10, metadata !6} ; [ DW_TAG_member ]
!13 = metadata !{metadata !"0xd\00y\003\0064\0064\0064\000", metadata !19, metadata !10, metadata !6} ; [ DW_TAG_member ]
!14 = metadata !{metadata !"0xd\00P2\008\00128\0064\00128\000", metadata !19, metadata !7, metadata !10} ; [ DW_TAG_member ]
!15 = metadata !{i32 11, i32 0, metadata !1, null}
!16 = metadata !{i32 12, i32 0, metadata !17, null}
!17 = metadata !{metadata !"0xb\0011\000\000", metadata !19, metadata !1} ; [ DW_TAG_lexical_block ]
!18 = metadata !{metadata !1}
!19 = metadata !{metadata !"b2.c", metadata !"/tmp/"}
!20 = metadata !{i32 0}
!21 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
