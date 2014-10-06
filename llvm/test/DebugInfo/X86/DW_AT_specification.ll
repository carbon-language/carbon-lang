; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; test that the DW_AT_specification is a back edge in the file.

; CHECK: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_MIPS_linkage_name {{.*}} "_ZN3foo3barEv"
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_specification {{.*}} "_ZN3foo3barEv"


@_ZZN3foo3barEvE1x = constant i32 0, align 4

define void @_ZN3foo3barEv()  {
entry:
  ret void, !dbg !25
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!28}

!0 = metadata !{metadata !"0x11\004\00clang version 3.0 ()\000\00\000\00\000", metadata !27, metadata !1, metadata !1, metadata !3, metadata !18,  metadata !1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x2e\00bar\00bar\00_ZN3foo3barEv\004\000\001\000\006\00256\000\004", metadata !6, null, metadata !7, null, void ()* @_ZN3foo3barEv, null, metadata !11, null} ; [ DW_TAG_subprogram ] [line 4] [def] [bar]
!6 = metadata !{metadata !"0x29", metadata !27} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null, metadata !9}
!9 = metadata !{metadata !"0xf\00\000\0064\0064\000\0064", i32 0, null, metadata !10} ; [ DW_TAG_pointer_type ]
!10 = metadata !{metadata !"0x13\00foo\001\000\000\000\004\000", metadata !27, null, null, null, null, null, null} ; [ DW_TAG_structure_type ] [foo] [line 1, size 0, align 0, offset 0] [decl] [from ]
!11 = metadata !{metadata !"0x2e\00bar\00bar\00_ZN3foo3barEv\002\000\000\000\006\00256\000\002", metadata !6, metadata !12, metadata !7, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ]
!12 = metadata !{metadata !"0x2\00foo\001\008\008\000\000\000", metadata !27, null, null, metadata !13, null, null} ; [ DW_TAG_class_type ]
!13 = metadata !{metadata !11}
!18 = metadata !{metadata !20}
!20 = metadata !{metadata !"0x34\00x\00x\00\005\001\001", metadata !5, metadata !6, metadata !21, i32* @_ZZN3foo3barEvE1x, null} ; [ DW_TAG_variable ]
!21 = metadata !{metadata !"0x26\00\000\000\000\000\000", null, null, metadata !22} ; [ DW_TAG_const_type ]
!22 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!25 = metadata !{i32 6, i32 1, metadata !26, null}
!26 = metadata !{metadata !"0xb\004\0017\000", metadata !6, metadata !5} ; [ DW_TAG_lexical_block ]
!27 = metadata !{metadata !"nsNativeAppSupportBase.ii", metadata !"/Users/espindola/mozilla-central/obj-x86_64-apple-darwin11.2.0/toolkit/library"}
!28 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
