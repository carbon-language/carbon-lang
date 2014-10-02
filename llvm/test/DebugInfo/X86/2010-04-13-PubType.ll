; RUN: llc -O0 -asm-verbose -mtriple=x86_64-macosx -generate-dwarf-pub-sections=Enable < %s | FileCheck %s
; CHECK-NOT: .asciz "X" ## External Name
; CHECK: .asciz "Y" ## External Name
; Test to check type with no definition is listed in pubtypes section.
%struct.X = type opaque
%struct.Y = type { i32 }

define i32 @foo(%struct.X* %x, %struct.Y* %y) nounwind ssp {
entry:
  %x_addr = alloca %struct.X*                     ; <%struct.X**> [#uses=1]
  %y_addr = alloca %struct.Y*                     ; <%struct.Y**> [#uses=1]
  %retval = alloca i32                            ; <i32*> [#uses=2]
  %0 = alloca i32                                 ; <i32*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata !{%struct.X** %x_addr}, metadata !0, metadata !{metadata !"0x102"}), !dbg !13
  store %struct.X* %x, %struct.X** %x_addr
  call void @llvm.dbg.declare(metadata !{%struct.Y** %y_addr}, metadata !14, metadata !{metadata !"0x102"}), !dbg !13
  store %struct.Y* %y, %struct.Y** %y_addr
  store i32 0, i32* %0, align 4, !dbg !13
  %1 = load i32* %0, align 4, !dbg !13            ; <i32> [#uses=1]
  store i32 %1, i32* %retval, align 4, !dbg !13
  br label %return, !dbg !13

return:                                           ; preds = %entry
  %retval1 = load i32* %retval, !dbg !13          ; <i32> [#uses=1]
  ret i32 %retval1, !dbg !15
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!20}

!0 = metadata !{metadata !"0x101\00x\007\000", metadata !1, metadata !2, metadata !7} ; [ DW_TAG_arg_variable ]
!1 = metadata !{metadata !"0x2e\00foo\00foo\00foo\007\000\001\000\006\000\000\007", metadata !18, metadata !2, metadata !4, null, i32 (%struct.X*, %struct.Y*)* @foo, null, null, null} ; [ DW_TAG_subprogram ]
!2 = metadata !{metadata !"0x29", metadata !18} ; [ DW_TAG_file_type ]
!3 = metadata !{metadata !"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\000\00\000\00\000", metadata !18, metadata !19, metadata !19, metadata !17, null,  null} ; [ DW_TAG_compile_unit ]
!4 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !18, metadata !2, null, metadata !5, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = metadata !{metadata !6, metadata !7, metadata !9}
!6 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !18, metadata !2} ; [ DW_TAG_base_type ]
!7 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", metadata !18, metadata !2, metadata !8} ; [ DW_TAG_pointer_type ]
!8 = metadata !{metadata !"0x13\00X\003\000\000\000\004\000", metadata !18, metadata !2, null, null, null, null, null} ; [ DW_TAG_structure_type ] [X] [line 3, size 0, align 0, offset 0] [decl] [from ]
!9 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", metadata !18, metadata !2, metadata !10} ; [ DW_TAG_pointer_type ]
!10 = metadata !{metadata !"0x13\00Y\004\0032\0032\000\000\000", metadata !18, metadata !2, null, metadata !11, null, null, null} ; [ DW_TAG_structure_type ] [Y] [line 4, size 32, align 32, offset 0] [def] [from ]
!11 = metadata !{metadata !12}
!12 = metadata !{metadata !"0xd\00x\005\0032\0032\000\000", metadata !18, metadata !10, metadata !6} ; [ DW_TAG_member ]
!13 = metadata !{i32 7, i32 0, metadata !1, null}
!14 = metadata !{metadata !"0x101\00y\007\000", metadata !1, metadata !2, metadata !9} ; [ DW_TAG_arg_variable ]
!15 = metadata !{i32 7, i32 0, metadata !16, null}
!16 = metadata !{metadata !"0xb\007\000\000", metadata !18, metadata !1} ; [ DW_TAG_lexical_block ]
!17 = metadata !{metadata !1}
!18 = metadata !{metadata !"a.c", metadata !"/tmp/"}
!19 = metadata !{i32 0}
!20 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
