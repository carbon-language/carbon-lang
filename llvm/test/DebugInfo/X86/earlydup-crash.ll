; RUN: llc %s -mtriple=i386-apple-macosx10.6.7 -o /dev/null

; This used to crash because early dup was not ignoring debug instructions.

%struct.cpp_dir = type { %struct.cpp_dir*, i8*, i32, i8, i8**, i8*, i8* (i8*, %struct.cpp_dir*)*, i64, i32, i8 }

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

define internal i8* @framework_construct_pathname(i8* %fname, %struct.cpp_dir* %dir) nounwind ssp {
entry:
  br i1 undef, label %bb33, label %bb

bb:                                               ; preds = %entry
  %tmp = icmp eq i32 undef, 0
  %tmp1 = add i32 0, 11
  call void @llvm.dbg.value(metadata !{i32 %tmp1}, i64 0, metadata !0, metadata !{metadata !"0x102"})
  br i1 undef, label %bb18, label %bb31.preheader

bb31.preheader:                                   ; preds = %bb19, %bb
  %tmp2 = getelementptr inbounds i8* %fname, i32 0
  br label %bb31

bb18:                                             ; preds = %bb
  %tmp3 = icmp eq i32 undef, 0
  br i1 %tmp3, label %bb19, label %bb33

bb19:                                             ; preds = %bb18
  call void @foobar(i32 0)
  br label %bb31.preheader

bb22:                                             ; preds = %bb31
  %tmp4 = add i32 0, %tmp1
  call void @foobar(i32 %tmp4)
  br i1 undef, label %bb33, label %bb31

bb31:                                             ; preds = %bb22, %bb31.preheader
  br i1 false, label %bb33, label %bb22

bb33:                                             ; preds = %bb31, %bb22, %bb18, %entry
  ret i8* undef
}

declare void @foobar(i32)

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!47}
!0 = metadata !{metadata !"0x100\00frname_len\00517\000", metadata !1, metadata !3, metadata !38} ; [ DW_TAG_auto_variable ]
!1 = metadata !{metadata !"0xb\00515\000\0019", metadata !44, metadata !2} ; [ DW_TAG_lexical_block ]
!2 = metadata !{metadata !"0x2e\00framework_construct_pathname\00framework_construct_pathname\00\00515\001\001\000\006\00256\001\000", metadata !44, null, metadata !5, null, i8* (i8*, %struct.cpp_dir*)* @framework_construct_pathname, null, null, null} ; [ DW_TAG_subprogram ]
!3 = metadata !{metadata !"0x29", metadata !44}  ; [ DW_TAG_file_type ]
!4 = metadata !{metadata !"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\001\00\000\00\000", metadata !44, metadata !46, metadata !46, metadata !45, null, null} ; [ DW_TAG_compile_unit ]
!5 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !44, metadata !3, null, metadata !6, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!6 = metadata !{metadata !7, metadata !9, metadata !11}
!7 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", metadata !44, metadata !3, metadata !8} ; [ DW_TAG_pointer_type ]
!8 = metadata !{metadata !"0x24\00char\000\008\008\000\000\006", metadata !44, metadata !3} ; [ DW_TAG_base_type ]
!9 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", metadata !44, metadata !3, metadata !10} ; [ DW_TAG_pointer_type ]
!10 = metadata !{metadata !"0x26\00\000\008\008\000\000", metadata !44, metadata !3, metadata !8} ; [ DW_TAG_const_type ]
!11 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", metadata !44, metadata !3, metadata !12} ; [ DW_TAG_pointer_type ]
!12 = metadata !{metadata !"0x16\00cpp_dir\0045\000\000\000\000", metadata !41, metadata !13, metadata !14} ; [ DW_TAG_typedef ]
!13 = metadata !{metadata !"0x29", metadata !41} ; [ DW_TAG_file_type ]
!14 = metadata !{metadata !"0x13\00cpp_dir\0043\00352\0032\000\000\000", metadata !41, metadata !3, null, metadata !15, null, null, null} ; [ DW_TAG_structure_type ] [cpp_dir] [line 43, size 352, align 32, offset 0] [def] [from ]
!15 = metadata !{metadata !16, metadata !18, metadata !19, metadata !21, metadata !23, metadata !25, metadata !27, metadata !29, metadata !33, metadata !36}
!16 = metadata !{metadata !"0xd\00next\00572\0032\0032\000\000", metadata !41, metadata !14, metadata !17} ; [ DW_TAG_member ]
!17 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", metadata !44, metadata !3, metadata !14} ; [ DW_TAG_pointer_type ]
!18 = metadata !{metadata !"0xd\00name\00575\0032\0032\0032\000", metadata !41, metadata !14, metadata !7} ; [ DW_TAG_member ]
!19 = metadata !{metadata !"0xd\00len\00576\0032\0032\0064\000", metadata !41, metadata !14, metadata !20} ; [ DW_TAG_member ]
!20 = metadata !{metadata !"0x24\00unsigned int\000\0032\0032\000\000\007", metadata !44, metadata !3} ; [ DW_TAG_base_type ]
!21 = metadata !{metadata !"0xd\00sysp\00580\008\008\0096\000", metadata !41, metadata !14, metadata !22} ; [ DW_TAG_member ]
!22 = metadata !{metadata !"0x24\00unsigned char\000\008\008\000\000\008", metadata !44, metadata !3} ; [ DW_TAG_base_type ]
!23 = metadata !{metadata !"0xd\00name_map\00584\0032\0032\00128\000", metadata !41, metadata !14, metadata !24} ; [ DW_TAG_member ]
!24 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", metadata !44, metadata !3, metadata !9} ; [ DW_TAG_pointer_type ]
!25 = metadata !{metadata !"0xd\00header_map\00590\0032\0032\00160\000", metadata !41, metadata !14, metadata !26} ; [ DW_TAG_member ]
!26 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", metadata !44, metadata !3, null} ; [ DW_TAG_pointer_type ]
!27 = metadata !{metadata !"0xd\00construct\00597\0032\0032\00192\000", metadata !41, metadata !14, metadata !28} ; [ DW_TAG_member ]
!28 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", metadata !44, metadata !3, metadata !5} ; [ DW_TAG_pointer_type ]
!29 = metadata !{metadata !"0xd\00ino\00601\0064\0064\00224\000", metadata !41, metadata !14, metadata !30} ; [ DW_TAG_member ]
!30 = metadata !{metadata !"0x16\00ino_t\00141\000\000\000\000", metadata !42, metadata !31, metadata !32} ; [ DW_TAG_typedef ]
!31 = metadata !{metadata !"0x29", metadata !42} ; [ DW_TAG_file_type ]
!32 = metadata !{metadata !"0x24\00long long unsigned int\000\0064\0064\000\000\007", metadata !44, metadata !3} ; [ DW_TAG_base_type ]
!33 = metadata !{metadata !"0xd\00dev\00602\0032\0032\00288\000", metadata !41, metadata !14, metadata !34} ; [ DW_TAG_member ]
!34 = metadata !{metadata !"0x16\00dev_t\00107\000\000\000\000", metadata !42, metadata !31, metadata !35} ; [ DW_TAG_typedef ]
!35 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !44, metadata !3} ; [ DW_TAG_base_type ]
!36 = metadata !{metadata !"0xd\00user_supplied_p\00605\008\008\00320\000", metadata !41, metadata !14, metadata !37} ; [ DW_TAG_member ]
!37 = metadata !{metadata !"0x24\00_Bool\000\008\008\000\000\002", metadata !44, metadata !3} ; [ DW_TAG_base_type ]
!38 = metadata !{metadata !"0x16\00size_t\00326\000\000\000\000", metadata !43, metadata !39, metadata !40} ; [ DW_TAG_typedef ]
!39 = metadata !{metadata !"0x29", metadata !43} ; [ DW_TAG_file_type ]
!40 = metadata !{metadata !"0x24\00long unsigned int\000\0032\0032\000\000\007", metadata !44, metadata !3} ; [ DW_TAG_base_type ]
!41 = metadata !{metadata !"cpplib.h", metadata !"/Users/espindola/llvm/build-llvm-gcc/gcc/../../llvm-gcc-4.2/gcc/../libcpp/include"}
!42 = metadata !{metadata !"types.h", metadata !"/usr/include/sys"}
!43 = metadata !{metadata !"stddef.h", metadata !"/Users/espindola/llvm/build-llvm-gcc/./prev-gcc/include"}
!44 = metadata !{metadata !"darwin-c.c", metadata !"/Users/espindola/llvm/build-llvm-gcc/gcc/../../llvm-gcc-4.2/gcc/config"}
!45 = metadata !{metadata !2}
!46 = metadata !{i32 0}
!47 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
