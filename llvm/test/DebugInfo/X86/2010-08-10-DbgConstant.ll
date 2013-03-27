; RUN: llc  -mtriple=i686-linux -O0 -filetype=obj -o %t %s
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
; CHECK: DW_TAG_constant [4]
; CHECK-NEXT: DW_AT_name [DW_FORM_strp] ( .debug_str[0x0000002c] = "ro")

define void @foo() nounwind ssp {
entry:
  call void @bar(i32 201), !dbg !8
  ret void, !dbg !8
}

declare void @bar(i32)

!llvm.dbg.cu = !{!2}

!0 = metadata !{i32 786478, metadata !1, null, metadata !"foo", metadata !"foo", metadata !"foo", metadata !1, i32 3, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, void ()* @foo, null, null, null, i32 3} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 786473, metadata !12, null} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 786449, metadata !12, null, i32 12, metadata !"clang 2.8", i1 false, metadata !"", i32 0, null, null, metadata !10, metadata !11, metadata !""} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 786453, metadata !1, null, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{null}
!5 = metadata !{i32 786471, i32 0, metadata !1, metadata !"ro", metadata !"ro", metadata !"ro", metadata !1, i32 1, metadata !6, i1 true, i1 true, i32 201, null} ; [ DW_TAG_constant ]
!6 = metadata !{i32 786470, metadata !1, null, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !7} ; [ DW_TAG_const_type ]
!7 = metadata !{i32 786468, metadata !1, null, metadata !"unsigned int", metadata !1, i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!8 = metadata !{i32 3, i32 14, metadata !9, null}
!9 = metadata !{i32 786443, metadata !0, null, i32 3, i32 12, metadata !1, i32 0} ; [ DW_TAG_lexical_block ]
!10 = metadata !{metadata !0}
!11 = metadata !{metadata !5}
!12 = metadata !{metadata !"/tmp/l.c", metadata !"/Volumes/Lalgate/clean/D"}
