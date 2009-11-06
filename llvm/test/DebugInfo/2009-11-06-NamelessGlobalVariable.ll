; RUN: llc %s -o /dev/null
@0 = internal constant i32 1                      ; <i32*> [#uses=1]

!llvm.dbg.gv = !{!0}

!0 = metadata !{i32 458804, i32 0, metadata !1, metadata !"", metadata !"", metadata !"", metadata !1, i32 378, metadata !2, i1 true, i1 true, i32* @0}; [DW_TAG_variable ]
!1 = metadata !{i32 458769, i32 0, i32 1, metadata !"cbdsqr.f", metadata !"/home/duncan/LLVM/dragonegg/unsolved/", metadata !"4.5.0 20091030 (experimental)", i1 true, i1 false, metadata !"", i32 0}; [DW_TAG_compile_unit ]
!2 = metadata !{i32 458788, metadata !1, metadata !"integer(kind=4)", metadata !1, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5}; [DW_TAG_base_type ]
