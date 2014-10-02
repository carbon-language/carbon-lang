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

!0 = metadata !{metadata !"0x11\0012\00clang version 3.1 (trunk 152837) (llvm/trunk 152845)\000\00\000\00\000", metadata !11, metadata !1, metadata !1, metadata !1, metadata !3,  metadata !1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x34\00f\00f\00\005\000\001", null, metadata !6, metadata !7, %struct.foo* @f, null} ; [ DW_TAG_variable ]
!6 = metadata !{metadata !"0x29", metadata !11} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x13\00foo\001\0032\0032\000\000\000", metadata !11, null, null, metadata !8, null, null, null} ; [ DW_TAG_structure_type ] [foo] [line 1, size 32, align 32, offset 0] [def] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{metadata !"0xd\00a\002\0032\0032\000\000", metadata !11, metadata !7, metadata !10} ; [ DW_TAG_member ]
!10 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!11 = metadata !{metadata !"struct_bug.c", metadata !"/Users/echristo/tmp"}
!12 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
