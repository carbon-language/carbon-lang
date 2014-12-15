; RUN: llc %s -o /dev/null
@0 = internal constant i32 1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9}

!0 = !{!"0x11\0012\00clang version 3.0 (trunk 139632)\001\00\000\00\000", !8, !2, !2, !2, !3, null} ; [ DW_TAG_compile_unit ]
!2 = !{}
!3 = !{!5}
!5 = !{!"0x34\00a\00a\00\002\000\001", null, !6, !7, i32* @0, null} ; [ DW_TAG_variable ]
!6 = !{!"0x29", !8} ; [ DW_TAG_file_type ]
!7 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!8 = !{!"g.c", !"/private/tmp"}
!9 = !{i32 1, !"Debug Info Version", i32 2}
