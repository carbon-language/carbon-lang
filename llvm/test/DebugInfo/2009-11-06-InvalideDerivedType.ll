; RUN: llc %s -o /dev/null
%struct._t = type { i32 }

@s1 = common global %struct._t zeroinitializer, align 4 ; <%struct._t*> [#uses=0]

!llvm.dbg.gv = !{!0}

!0 = metadata !{i32 458804, i32 0, metadata !1, metadata !"s1", metadata !"s1", metadata !"s1", metadata !1, i32 3, metadata !2, i1 false, i1 true, %struct._t* @s1}; [DW_TAG_variable ]
!1 = metadata !{i32 458769, i32 0, i32 12, metadata !"t.c", metadata !"/tmp", metadata !"clang 1.1", i1 true, i1 false, metadata !"", i32 0}; [DW_TAG_compile_unit ]
!2 = metadata !{i32 458771, metadata !1, metadata !"_t", metadata !1, i32 1, i64 32, i64 32, i64 0, i32 0, null, metadata !3, i32 0}; [DW_TAG_structure_type ]
!3 = metadata !{metadata !4}
!4 = metadata !{i32 458765, metadata !1, metadata !"j", metadata !1, i32 2, i64 32, i64 32, i64 0, i32 0, metadata !5}; [DW_TAG_member ]
!5 = metadata !{i32 458790, metadata !1, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, null}; [DW_TAG_const_type ]
