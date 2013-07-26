; RUN: llc -O0 -relocation-model pic < %s -o /dev/null
; PR7545
@.str = private constant [4 x i8] c"one\00", align 1 ; <[4 x i8]*> [#uses=1]
@.str1 = private constant [4 x i8] c"two\00", align 1 ; <[5 x i8]*> [#uses=1]
@C.9.2167 = internal constant [2 x i8*] [i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8]* @.str1, i64 0, i64 0)]
!38 = metadata !{i32 524329, metadata !109} ; [ DW_TAG_file_type ]
!39 = metadata !{i32 524305, metadata !109, i32 1, metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build 9999)", i1 true, metadata !"", i32 0, metadata !108, metadata !108, null, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!46 = metadata !{i32 524303, metadata !109, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !47} ; [ DW_TAG_pointer_type ]
!47 = metadata !{i32 524324, metadata !109, null, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
!97 = metadata !{i32 524334, i32 0, metadata !39, metadata !"main", metadata !"main", metadata !"main", i32 73, metadata !98, i1 false, i1 true,
                i32 0, i32 0, null, i32 0, i32 0, null, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!98 = metadata !{i32 524309, metadata !109, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !99, i32 0, null} ; [ DW_TAG_subroutine_type ]
!99 = metadata !{metadata !100}
!100 = metadata !{i32 524324, metadata !109, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!101 = metadata !{[2 x i8*]* @C.9.2167}
!102 = metadata !{i32 524544, metadata !103, metadata !"find_strings", metadata !38, i32 75, metadata !104, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!103 = metadata !{i32 524299, null, metadata !97, i32 73, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
!104 = metadata !{i32 524289, metadata !109, null, metadata !"", i32 0, i64 85312, i64 64, i64 0, i32 0, metadata !46, metadata !105, i32 0, null} ; [ DW_TAG_array_type ]
!105 = metadata !{metadata !106}
!106 = metadata !{i32 524321, i64 0, i64 1333}    ; [ DW_TAG_subrange_type ]
!107 = metadata !{i32 73, i32 0, metadata !103, null}
!108 = metadata !{i32 0}
!109 = metadata !{metadata !"pbmsrch.c", metadata !"/Users/grawp/LLVM/test-suite/MultiSource/Benchmarks/MiBench/office-stringsearch"}

define i32 @main() nounwind ssp {
bb.nph:
  tail call void @llvm.dbg.declare(metadata !101, metadata !102), !dbg !107
  ret i32 0, !dbg !107
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

