; RUN: llc < %s | grep low_pc | count 2
@i = global i32 1                                 ; <i32*> [#uses=0]

!llvm.dbg.gv = !{!0}

!0 = metadata !{i32 524340, i32 0, metadata !1, metadata !"i", metadata !"i", metadata !"", metadata !1, i32 1, metadata !3, i1 false, i1 true, i32* @i} ; [ DW_TAG_variable ]
!1 = metadata !{i32 524329, metadata !"b.c", metadata !"/tmp", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 524305, i32 0, i32 1, metadata !"b.c", metadata !"/tmp", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 524324, metadata !1, metadata !"int", metadata !1, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
