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
  call void @llvm.dbg.value(metadata i32 %tmp1, i64 0, metadata !0, metadata !{!"0x102"})
  br i1 undef, label %bb18, label %bb31.preheader

bb31.preheader:                                   ; preds = %bb19, %bb
  %tmp2 = getelementptr inbounds i8, i8* %fname, i32 0
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
!0 = !{!"0x100\00frname_len\00517\000", !1, !3, !38} ; [ DW_TAG_auto_variable ]
!1 = !{!"0xb\00515\000\0019", !44, !2} ; [ DW_TAG_lexical_block ]
!2 = !{!"0x2e\00framework_construct_pathname\00framework_construct_pathname\00\00515\001\001\000\006\00256\001\000", !44, null, !5, null, i8* (i8*, %struct.cpp_dir*)* @framework_construct_pathname, null, null, null} ; [ DW_TAG_subprogram ]
!3 = !{!"0x29", !44}  ; [ DW_TAG_file_type ]
!4 = !{!"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\001\00\000\00\000", !44, !46, !46, !45, null, null} ; [ DW_TAG_compile_unit ]
!5 = !{!"0x15\00\000\000\000\000\000\000", !44, !3, null, !6, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!6 = !{!7, !9, !11}
!7 = !{!"0xf\00\000\0032\0032\000\000", !44, !3, !8} ; [ DW_TAG_pointer_type ]
!8 = !{!"0x24\00char\000\008\008\000\000\006", !44, !3} ; [ DW_TAG_base_type ]
!9 = !{!"0xf\00\000\0032\0032\000\000", !44, !3, !10} ; [ DW_TAG_pointer_type ]
!10 = !{!"0x26\00\000\008\008\000\000", !44, !3, !8} ; [ DW_TAG_const_type ]
!11 = !{!"0xf\00\000\0032\0032\000\000", !44, !3, !12} ; [ DW_TAG_pointer_type ]
!12 = !{!"0x16\00cpp_dir\0045\000\000\000\000", !41, !13, !14} ; [ DW_TAG_typedef ]
!13 = !{!"0x29", !41} ; [ DW_TAG_file_type ]
!14 = !{!"0x13\00cpp_dir\0043\00352\0032\000\000\000", !41, !3, null, !15, null, null, null} ; [ DW_TAG_structure_type ] [cpp_dir] [line 43, size 352, align 32, offset 0] [def] [from ]
!15 = !{!16, !18, !19, !21, !23, !25, !27, !29, !33, !36}
!16 = !{!"0xd\00next\00572\0032\0032\000\000", !41, !14, !17} ; [ DW_TAG_member ]
!17 = !{!"0xf\00\000\0032\0032\000\000", !44, !3, !14} ; [ DW_TAG_pointer_type ]
!18 = !{!"0xd\00name\00575\0032\0032\0032\000", !41, !14, !7} ; [ DW_TAG_member ]
!19 = !{!"0xd\00len\00576\0032\0032\0064\000", !41, !14, !20} ; [ DW_TAG_member ]
!20 = !{!"0x24\00unsigned int\000\0032\0032\000\000\007", !44, !3} ; [ DW_TAG_base_type ]
!21 = !{!"0xd\00sysp\00580\008\008\0096\000", !41, !14, !22} ; [ DW_TAG_member ]
!22 = !{!"0x24\00unsigned char\000\008\008\000\000\008", !44, !3} ; [ DW_TAG_base_type ]
!23 = !{!"0xd\00name_map\00584\0032\0032\00128\000", !41, !14, !24} ; [ DW_TAG_member ]
!24 = !{!"0xf\00\000\0032\0032\000\000", !44, !3, !9} ; [ DW_TAG_pointer_type ]
!25 = !{!"0xd\00header_map\00590\0032\0032\00160\000", !41, !14, !26} ; [ DW_TAG_member ]
!26 = !{!"0xf\00\000\0032\0032\000\000", !44, !3, null} ; [ DW_TAG_pointer_type ]
!27 = !{!"0xd\00construct\00597\0032\0032\00192\000", !41, !14, !28} ; [ DW_TAG_member ]
!28 = !{!"0xf\00\000\0032\0032\000\000", !44, !3, !5} ; [ DW_TAG_pointer_type ]
!29 = !{!"0xd\00ino\00601\0064\0064\00224\000", !41, !14, !30} ; [ DW_TAG_member ]
!30 = !{!"0x16\00ino_t\00141\000\000\000\000", !42, !31, !32} ; [ DW_TAG_typedef ]
!31 = !{!"0x29", !42} ; [ DW_TAG_file_type ]
!32 = !{!"0x24\00long long unsigned int\000\0064\0064\000\000\007", !44, !3} ; [ DW_TAG_base_type ]
!33 = !{!"0xd\00dev\00602\0032\0032\00288\000", !41, !14, !34} ; [ DW_TAG_member ]
!34 = !{!"0x16\00dev_t\00107\000\000\000\000", !42, !31, !35} ; [ DW_TAG_typedef ]
!35 = !{!"0x24\00int\000\0032\0032\000\000\005", !44, !3} ; [ DW_TAG_base_type ]
!36 = !{!"0xd\00user_supplied_p\00605\008\008\00320\000", !41, !14, !37} ; [ DW_TAG_member ]
!37 = !{!"0x24\00_Bool\000\008\008\000\000\002", !44, !3} ; [ DW_TAG_base_type ]
!38 = !{!"0x16\00size_t\00326\000\000\000\000", !43, !39, !40} ; [ DW_TAG_typedef ]
!39 = !{!"0x29", !43} ; [ DW_TAG_file_type ]
!40 = !{!"0x24\00long unsigned int\000\0032\0032\000\000\007", !44, !3} ; [ DW_TAG_base_type ]
!41 = !{!"cpplib.h", !"/Users/espindola/llvm/build-llvm-gcc/gcc/../../llvm-gcc-4.2/gcc/../libcpp/include"}
!42 = !{!"types.h", !"/usr/include/sys"}
!43 = !{!"stddef.h", !"/Users/espindola/llvm/build-llvm-gcc/./prev-gcc/include"}
!44 = !{!"darwin-c.c", !"/Users/espindola/llvm/build-llvm-gcc/gcc/../../llvm-gcc-4.2/gcc/config"}
!45 = !{!2}
!46 = !{i32 0}
!47 = !{i32 1, !"Debug Info Version", i32 2}
