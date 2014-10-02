; RUN: llc -O0 -relocation-model pic < %s -o /dev/null
; PR7545
@.str = private constant [4 x i8] c"one\00", align 1 ; <[4 x i8]*> [#uses=1]
@.str1 = private constant [4 x i8] c"two\00", align 1 ; <[5 x i8]*> [#uses=1]
@C.9.2167 = internal constant [2 x i8*] [i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8]* @.str1, i64 0, i64 0)]
!38 = metadata !{metadata !"0x29", metadata !109} ; [ DW_TAG_file_type ]
!39 = metadata !{metadata !"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build 9999)\001\00\000\00\000", metadata !109, metadata !108, metadata !108, null, null, null} ; [ DW_TAG_compile_unit ]
!46 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", metadata !109, null, metadata !47} ; [ DW_TAG_pointer_type ]
!47 = metadata !{metadata !"0x24\00char\000\008\008\000\000\006", metadata !109, null} ; [ DW_TAG_base_type ]
!97 = metadata !{metadata !"0x2e\00main\00main\00main\0073\000\001\000\006\000\000\000", i32 0, metadata !39, metadata !98, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!98 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !109, null, null, metadata !99, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!99 = metadata !{metadata !100}
!100 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !109, null} ; [ DW_TAG_base_type ]
!101 = metadata !{[2 x i8*]* @C.9.2167}
!102 = metadata !{metadata !"0x100\00find_strings\0075\000", metadata !103, metadata !38, metadata !104} ; [ DW_TAG_auto_variable ]
!103 = metadata !{metadata !"0xb\0073\000\000", null, metadata !97} ; [ DW_TAG_lexical_block ]
!104 = metadata !{metadata !"0x1\00\000\0085312\0064\000\000", metadata !109, null, metadata !46, metadata !105, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 85312, align 64, offset 0] [from ]
!105 = metadata !{metadata !106}
!106 = metadata !{metadata !"0x21\000\001333"}    ; [ DW_TAG_subrange_type ]
!107 = metadata !{i32 73, i32 0, metadata !103, null}
!108 = metadata !{i32 0}
!109 = metadata !{metadata !"pbmsrch.c", metadata !"/Users/grawp/LLVM/test-suite/MultiSource/Benchmarks/MiBench/office-stringsearch"}

define i32 @main() nounwind ssp {
bb.nph:
  tail call void @llvm.dbg.declare(metadata !101, metadata !102, metadata !{metadata !"0x102"}), !dbg !107
  ret i32 0, !dbg !107
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

