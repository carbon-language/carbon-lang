; RUN: %llc_dwarf -O0 < %s | grep AT_decl_file |  grep 2
; Here _ZN1S3fooEv is defined in header file identified as AT_decl_file no. 2 in debug info.
%struct.S = type <{ i8 }>

define i32 @_Z3barv() nounwind ssp {
entry:
  %retval = alloca i32                            ; <i32*> [#uses=2]
  %0 = alloca i32                                 ; <i32*> [#uses=2]
  %s1 = alloca %struct.S                          ; <%struct.S*> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata !{%struct.S* %s1}, metadata !0, metadata !{metadata !"0x102"}), !dbg !16
  %1 = call i32 @_ZN1S3fooEv(%struct.S* %s1) nounwind, !dbg !17 ; <i32> [#uses=1]
  store i32 %1, i32* %0, align 4, !dbg !17
  %2 = load i32* %0, align 4, !dbg !17            ; <i32> [#uses=1]
  store i32 %2, i32* %retval, align 4, !dbg !17
  br label %return, !dbg !17

return:                                           ; preds = %entry
  %retval1 = load i32* %retval, !dbg !17          ; <i32> [#uses=1]
  ret i32 %retval1, !dbg !16
}

define linkonce_odr i32 @_ZN1S3fooEv(%struct.S* %this) nounwind ssp align 2 {
entry:
  %this_addr = alloca %struct.S*                  ; <%struct.S**> [#uses=1]
  %retval = alloca i32                            ; <i32*> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata !{%struct.S** %this_addr}, metadata !18, metadata !{metadata !"0x102"}), !dbg !21
  store %struct.S* %this, %struct.S** %this_addr
  br label %return, !dbg !21

return:                                           ; preds = %entry
  %retval1 = load i32* %retval, !dbg !21          ; <i32> [#uses=1]
  ret i32 %retval1, !dbg !22
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!28}

!0 = metadata !{metadata !"0x100\00s1\003\000", metadata !1, metadata !4, metadata !9} ; [ DW_TAG_auto_variable ]
!1 = metadata !{metadata !"0xb\003\000\000", metadata !25, metadata !2} ; [ DW_TAG_lexical_block ]
!2 = metadata !{metadata !"0xb\003\000\000", metadata !25, metadata !3} ; [ DW_TAG_lexical_block ]
!3 = metadata !{metadata !"0x2e\00bar\00bar\00_Z3barv\003\000\001\000\006\000\000\003", metadata !25, metadata !4, metadata !6, null, i32 ()* @_Z3barv, null, null, null} ; [ DW_TAG_subprogram ]
!4 = metadata !{metadata !"0x29", metadata !25} ; [ DW_TAG_file_type ]
!5 = metadata !{metadata !"0x11\004\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\000\00\000\00\000", metadata !25, metadata !27, metadata !27, metadata !24, null,  null} ; [ DW_TAG_compile_unit ]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !25, metadata !4, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8}
!8 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !25, metadata !4} ; [ DW_TAG_base_type ]
!9 = metadata !{metadata !"0x13\00S\002\008\008\000\000\000", metadata !26, metadata !4, null, metadata !11, null, null, null} ; [ DW_TAG_structure_type ] [S] [line 2, size 8, align 8, offset 0] [def] [from ]
!10 = metadata !{metadata !"0x29", metadata !26} ; [ DW_TAG_file_type ]
!11 = metadata !{metadata !12}
!12 = metadata !{metadata !"0x2e\00foo\00foo\00_ZN1S3fooEv\003\000\001\000\006\000\000\003", metadata !26, metadata !9, metadata !13, null, i32 (%struct.S*)* @_ZN1S3fooEv, null, null, null} ; [ DW_TAG_subprogram ]
!13 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !25, null, null, metadata !14, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = metadata !{metadata !8, metadata !15}
!15 = metadata !{metadata !"0xf\00\000\0064\0064\000\0064", metadata !25, metadata !4, metadata !9} ; [ DW_TAG_pointer_type ]
!16 = metadata !{i32 3, i32 0, metadata !1, null}
!17 = metadata !{i32 3, i32 0, metadata !3, null}
!18 = metadata !{metadata !"0x101\00this\003\000", metadata !12, metadata !10, metadata !19} ; [ DW_TAG_arg_variable ]
!19 = metadata !{metadata !"0x26\00\000\0064\0064\000\0064", metadata !25, metadata !4, metadata !20} ; [ DW_TAG_const_type ]
!20 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", metadata !25, metadata !4, metadata !9} ; [ DW_TAG_pointer_type ]
!21 = metadata !{i32 3, i32 0, metadata !12, null}
!22 = metadata !{i32 3, i32 0, metadata !23, null}
!23 = metadata !{metadata !"0xb\003\000\000", metadata !26, metadata !12} ; [ DW_TAG_lexical_block ]
!24 = metadata !{metadata !3, metadata !12}
!25 = metadata !{metadata !"one.cc", metadata !"/tmp/"}
!26 = metadata !{metadata !"one.h", metadata !"/tmp/"}
!27 = metadata !{i32 0}
!28 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
