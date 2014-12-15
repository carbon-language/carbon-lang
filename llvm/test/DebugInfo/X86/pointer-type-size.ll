; RUN: llc -mtriple=x86_64-apple-macosx10.7 %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; CHECK: ptr
; CHECK-NOT: AT_bit_size

%struct.crass = type { i8* }

@crass = common global %struct.crass zeroinitializer, align 8

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14}

!0 = !{!"0x11\0012\00clang version 3.1 (trunk 147882)\000\00\000\00\000", !13, !1, !1, !1, !3,  !1} ; [ DW_TAG_compile_unit ]
!1 = !{}
!3 = !{!5}
!5 = !{!"0x34\00crass\00crass\00\001\000\001", null, !6, !7, %struct.crass* @crass, null} ; [ DW_TAG_variable ]
!6 = !{!"0x29", !13} ; [ DW_TAG_file_type ]
!7 = !{!"0x13\00crass\001\0064\0064\000\000\000", !13, null, null, !8, null, null, null} ; [ DW_TAG_structure_type ] [crass] [line 1, size 64, align 64, offset 0] [def] [from ]
!8 = !{!9}
!9 = !{!"0xd\00ptr\001\0064\0064\000\000", !13, !7, !10} ; [ DW_TAG_member ]
!10 = !{!"0x26\00\000\000\000\000\000", null, null, !11} ; [ DW_TAG_const_type ]
!11 = !{!"0xf\00\000\0064\0064\000\000", null, null, !12} ; [ DW_TAG_pointer_type ]
!12 = !{!"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ]
!13 = !{!"foo.c", !"/Users/echristo/tmp"}
!14 = !{i32 1, !"Debug Info Version", i32 2}
