; RUN: llc  < %s | FileCheck %s
; Radar 10147769
; Do not unnecessarily use AT_specification DIE.
; CHECK-NOT: AT_specification

@a = common global [10 x i32] zeroinitializer, align 16

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12}

!0 = !{!"0x11\0012\00clang version 3.0 (trunk 140253)\001\00\000\00\000", !11, !2, !2, !2, !3, null} ; [ DW_TAG_compile_unit ]
!2 = !{}
!3 = !{!5}
!5 = !{!"0x34\00a\00a\00\001\000\001", null, !6, !7, [10 x i32]* @a, null} ; [ DW_TAG_variable ]
!6 = !{!"0x29", !11} ; [ DW_TAG_file_type ]
!7 = !{!"0x1\00\000\00320\0032\000\000", null, null, !8, !9, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 320, align 32, offset 0] [from int]
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!9 = !{!10}
!10 = !{!"0x21\000\0010"}        ; [ DW_TAG_subrange_type ]
!11 = !{!"x.c", !"/private/tmp"}
!12 = !{i32 1, !"Debug Info Version", i32 2}
