; RUN: llc -O0 -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump %t | FileCheck %s

@e = global i16 0, align 2

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 786449, i32 0, i32 4, metadata !"foo.cpp", metadata !"/Users/echristo/tmp", metadata !"clang version 3.2 (trunk 157772) (llvm/trunk 157761)", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !6, metadata !6, metadata !7} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{metadata !3}
!3 = metadata !{i32 786436, null, metadata !"E", metadata !4, i32 1, i64 16, i64 16, i32 0, i32 4, null, metadata !5, i32 0, i32 0} ; [ DW_TAG_enumeration_type ]
!4 = metadata !{i32 786473, metadata !"foo.cpp", metadata !"/Users/echristo/tmp", null} ; [ DW_TAG_file_type ]
!5 = metadata !{i32 0}
!6 = metadata !{metadata !5}
!7 = metadata !{metadata !8}
!8 = metadata !{metadata !9}
!9 = metadata !{i32 786484, i32 0, null, metadata !"e", metadata !"e", metadata !"", metadata !4, i32 2, metadata !3, i32 0, i32 1, i16* @e} ; [ DW_TAG_variable ]

; CHECK: DW_TAG_enumeration_type
; CHECK-NEXT: DW_AT_name
; CHECK-NEXT: DW_AT_byte_size
; CHECK-NEXT: DW_AT_declaration
