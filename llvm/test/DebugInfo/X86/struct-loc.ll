; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; Make sure that structures have a decl file and decl line attached.
; CHECK: DW_TAG_structure_type
; CHECK: DW_AT_decl_file
; CHECK: DW_AT_decl_line
; CHECK: DW_TAG_member

%struct.foo = type { i32 }

@f = common global %struct.foo zeroinitializer, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12}

!0 = !{!"0x11\0012\00clang version 3.1 (trunk 152837) (llvm/trunk 152845)\000\00\000\00\000", !11, !1, !1, !1, !3,  !1} ; [ DW_TAG_compile_unit ]
!1 = !{}
!3 = !{!5}
!5 = !{!"0x34\00f\00f\00\005\000\001", null, !6, !7, %struct.foo* @f, null} ; [ DW_TAG_variable ]
!6 = !{!"0x29", !11} ; [ DW_TAG_file_type ]
!7 = !{!"0x13\00foo\001\0032\0032\000\000\000", !11, null, null, !8, null, null, null} ; [ DW_TAG_structure_type ] [foo] [line 1, size 32, align 32, offset 0] [def] [from ]
!8 = !{!9}
!9 = !{!"0xd\00a\002\0032\0032\000\000", !11, !7, !10} ; [ DW_TAG_member ]
!10 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!11 = !{!"struct_bug.c", !"/Users/echristo/tmp"}
!12 = !{i32 1, !"Debug Info Version", i32 2}
