; RUN: llc %s -o /dev/null
@0 = internal constant i32 1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.0 (trunk 139632)\001\00\000\00\000", metadata !8, metadata !2, metadata !2, metadata !2, metadata !3, null} ; [ DW_TAG_compile_unit ]
!2 = metadata !{}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x34\00a\00a\00\002\000\001", null, metadata !6, metadata !7, i32* @0, null} ; [ DW_TAG_variable ]
!6 = metadata !{metadata !"0x29", metadata !8} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!8 = metadata !{metadata !"g.c", metadata !"/private/tmp"}
!9 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
