; RUN: llc -mtriple=x86_64-apple-darwin < %s -o -

; PR16954
;
; Make sure that when we splice off the end of a machine basic block, we include
; DBG_VALUE MI in the terminator sequence.

@a = external global { i64, [56 x i8] }, align 32

; Function Attrs: nounwind sspreq
define i32 @_Z18read_response_sizev() #0 {
entry:
  tail call void @llvm.dbg.value(metadata !22, i64 0, metadata !23, metadata !{metadata !"0x102"}), !dbg !39
  %0 = load i64* getelementptr inbounds ({ i64, [56 x i8] }* @a, i32 0, i32 0), align 8, !dbg !40
  tail call void @llvm.dbg.value(metadata !63, i64 0, metadata !64, metadata !{metadata !"0x102"}), !dbg !71
  %1 = trunc i64 %0 to i32
  ret i32 %1
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

attributes #0 = { sspreq }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !72}

!0 = metadata !{metadata !"0x11\004\00clang version 3.4 \001\00\000\00\001", metadata !1, metadata !2, metadata !5, metadata !8, metadata !20, metadata !5} ; [ DW_TAG_compile_unit ] [/Users/matt/ryan_bug/<unknown>] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"<unknown>", metadata !"/Users/matt/ryan_bug"}
!2 = metadata !{metadata !3}
!3 = metadata !{metadata !"0x4\00\0020\0032\0032\000\000\000", metadata !1, metadata !4, null, metadata !6, null, null, null} ; [ DW_TAG_enumeration_type ] [line 20, size 32, align 32, offset 0] [def] [from ]
!4 = metadata !{metadata !"0x13\00C\0019\008\008\000\000\000", metadata !1, null, null, metadata !5, null, null, null} ; [ DW_TAG_structure_type ] [C] [line 19, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{}
!6 = metadata !{metadata !7}
!7 = metadata !{metadata !"0x28\00max_frame_size\000"} ; [ DW_TAG_enumerator ] [max_frame_size :: 0]
!8 = metadata !{metadata !9, metadata !24, metadata !41, metadata !65}
!9 = metadata !{metadata !"0x2e\00read_response_size\00read_response_size\00_Z18read_response_sizev\0027\000\001\000\006\00256\001\0027", metadata !1, metadata !10, metadata !11, null, i32 ()* @_Z18read_response_sizev, null, null, metadata !14} ; [ DW_TAG_subprogram ] [line 27] [def] [read_response_size]
!10 = metadata !{metadata !"0x29", metadata !1}         ; [ DW_TAG_file_type ] [/Users/matt/ryan_bug/<unknown>]
!11 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{metadata !13}
!13 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!14 = metadata !{metadata !15, metadata !19}
!15 = metadata !{metadata !"0x100\00b\0028\000", metadata !9, metadata !10, metadata !16} ; [ DW_TAG_auto_variable ] [b] [line 28]
!16 = metadata !{metadata !"0x13\00B\0016\0032\0032\000\000\000", metadata !1, null, null, metadata !17, null, null} ; [ DW_TAG_structure_type ] [B] [line 16, size 32, align 32, offset 0] [def] [from ]
!17 = metadata !{metadata !18}
!18 = metadata !{metadata !"0xd\00end_of_file\0017\0032\0032\000\000", metadata !1, metadata !16, metadata !13} ; [ DW_TAG_member ] [end_of_file] [line 17, size 32, align 32, offset 0] [from int]
!19 = metadata !{metadata !"0x100\00c\0029\000", metadata !9, metadata !10, metadata !13} ; [ DW_TAG_auto_variable ] [c] [line 29]
!20 = metadata !{}
!21 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!22 = metadata !{i64* getelementptr inbounds ({ i64, [56 x i8] }* @a, i32 0, i32 0)}
!23 = metadata !{metadata !"0x101\00p2\0033554444\000", metadata !24, metadata !10, metadata !32, metadata !38} ; [ DW_TAG_arg_variable ] [p2] [line 12]
!24 = metadata !{metadata !"0x2e\00min<unsigned long long>\00min<unsigned long long>\00_ZN3__13minIyEERKT_S3_RS1_\0012\000\001\000\006\00256\001\0012", metadata !1, metadata !25, metadata !27, null, null, metadata !33, null, metadata !35} ; [ DW_TAG_subprogram ] [line 12] [def] [min<unsigned long long>]
!25 = metadata !{metadata !"0x39\00__1\001", metadata !26, null} ; [ DW_TAG_namespace ] [__1] [line 1]
!26 = metadata !{metadata !"main.cpp", metadata !"/Users/matt/ryan_bug"}
!27 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !28, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!28 = metadata !{metadata !29, metadata !29, metadata !32}
!29 = metadata !{metadata !"0x10\00\000\000\000\000\000", null, null, metadata !30} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!30 = metadata !{metadata !"0x26\00\000\000\000\000\000", null, null, metadata !31} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from long long unsigned int]
!31 = metadata !{metadata !"0x24\00long long unsigned int\000\0064\0064\000\000\007", null, null} ; [ DW_TAG_base_type ] [long long unsigned int] [line 0, size 64, align 64, offset 0, enc DW_ATE_unsigned]
!32 = metadata !{metadata !"0x10\00\000\000\000\000\000", null, null, metadata !31} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from long long unsigned int]
!33 = metadata !{metadata !34}
!34 = metadata !{metadata !"0x2f\00_Tp\000\000", null, metadata !31, null} ; [ DW_TAG_template_type_parameter ]
!35 = metadata !{metadata !36, metadata !37}
!36 = metadata !{metadata !"0x101\00p1\0016777228\000", metadata !24, metadata !10, metadata !29} ; [ DW_TAG_arg_variable ] [p1] [line 12]
!37 = metadata !{metadata !"0x101\00p2\0033554444\000", metadata !24, metadata !10, metadata !32} ; [ DW_TAG_arg_variable ] [p2] [line 12]
!38 = metadata !{i32 33, i32 0, metadata !9, null}
!39 = metadata !{i32 12, i32 0, metadata !24, metadata !38}
!40 = metadata !{i32 9, i32 0, metadata !41, metadata !59}
!41 = metadata !{metadata !"0x2e\00min<unsigned long long, __1::A>\00min<unsigned long long, __1::A>\00_ZN3__13minIyNS_1AEEERKT_S4_RS2_T0_\007\000\001\000\006\00256\001\008", metadata !1, metadata !25, metadata !42, null, null, metadata !53, null, metadata !55} ; [ DW_TAG_subprogram ] [line 7] [def] [scope 8] [min<unsigned long long, __1::A>]
!42 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !43, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!43 = metadata !{metadata !29, metadata !29, metadata !32, metadata !44}
!44 = metadata !{metadata !"0x13\00A\000\008\008\000\000\000", metadata !1, metadata !25, null, metadata !45, null, null, null} ; [ DW_TAG_structure_type ] [A] [line 0, size 8, align 8, offset 0] [def] [from ]
!45 = metadata !{metadata !46}
!46 = metadata !{metadata !"0x2e\00operator()\00operator()\00_ZN3__11AclERKiS2_\001\000\000\000\006\00256\001\001", metadata !1, metadata !44, metadata !47, null, null, null, i32 0, metadata !52} ; [ DW_TAG_subprogram ] [line 1] [operator()]
!47 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !48, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!48 = metadata !{metadata !13, metadata !49, metadata !50, metadata !50}
!49 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088", i32 0, null, metadata !44} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from A]
!50 = metadata !{metadata !"0x10\00\000\000\000\000\000", null, null, metadata !51} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!51 = metadata !{metadata !"0x26\00\000\000\000\000\000", null, null, metadata !13} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from int]
!52 = metadata !{i32 786468}
!53 = metadata !{metadata !34, metadata !54}
!54 = metadata !{metadata !"0x2f\00_Compare\000\000", null, metadata !44, null} ; [ DW_TAG_template_type_parameter ]
!55 = metadata !{metadata !56, metadata !57, metadata !58}
!56 = metadata !{metadata !"0x101\00p1\0016777223\000", metadata !41, metadata !10, metadata !29} ; [ DW_TAG_arg_variable ] [p1] [line 7]
!57 = metadata !{metadata !"0x101\00p2\0033554439\000", metadata !41, metadata !10, metadata !32} ; [ DW_TAG_arg_variable ] [p2] [line 7]
!58 = metadata !{metadata !"0x101\00p3\0050331656\000", metadata !41, metadata !10, metadata !44} ; [ DW_TAG_arg_variable ] [p3] [line 8]
!59 = metadata !{i32 13, i32 0, metadata !24, metadata !38}
!63 = metadata !{i32 undef}
!64 = metadata !{metadata !"0x101\00p1\0033554433\000", metadata !65, metadata !10, metadata !50, metadata !40} ; [ DW_TAG_arg_variable ] [p1] [line 1]
!65 = metadata !{metadata !"0x2e\00operator()\00operator()\00_ZN3__11AclERKiS2_\001\000\001\000\006\00256\001\002", metadata !1, metadata !25, metadata !47, null, null, null, metadata !46, metadata !66} ; [ DW_TAG_subprogram ] [line 1] [def] [scope 2] [operator()]
!66 = metadata !{metadata !67, metadata !69, metadata !70}
!67 = metadata !{metadata !"0x101\00this\0016777216\001088", metadata !65, null, metadata !68} ; [ DW_TAG_arg_variable ] [this] [line 0]
!68 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !44} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from A]
!69 = metadata !{metadata !"0x101\00p1\0033554433\000", metadata !65, metadata !10, metadata !50} ; [ DW_TAG_arg_variable ] [p1] [line 1]
!70 = metadata !{metadata !"0x101\00\0050331650\000", metadata !65, metadata !10, metadata !50} ; [ DW_TAG_arg_variable ] [line 2]
!71 = metadata !{i32 1, i32 0, metadata !65, metadata !40}
!72 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
