; REQUIRES: object-emission

; RUN: %llc_dwarf -filetype=obj -O0 < %s > %t
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
; CHECK: DW_TAG_ptr_to_member_type
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4]       (cu + {{.*}} => {[[TYPE:0x[0-9a-f]+]]})
; CHECK: [[TYPE]]:   DW_TAG_base_type
; IR generated from clang -g with the following source:
; struct Foo {
;   int e;
; };
; int Foo:*x = 0;

@x = global i64 -1, align 8

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}

!0 = metadata !{metadata !"0x11\004\00clang version 3.4\000\00\000\00\000", metadata !1, metadata !2, metadata !3, metadata !2, metadata !5, metadata !2} ; [ DW_TAG_compile_unit ] [foo.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"foo.cpp", metadata !"."}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x13\00Foo\001\000\000\000\004\000", metadata !1, null, null, null, null, null, metadata !"_ZTS3Foo"} ; [ DW_TAG_structure_type ] [Foo] [line 1, size 0, align 0, offset 0] [decl] [from ]
!5 = metadata !{metadata !6}
!6 = metadata !{metadata !"0x34\00x\00x\00\004\000\001", null, metadata !7, metadata !8, i64* @x, null} ; [ DW_TAG_variable ] [x] [line 4] [def]
!7 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [foo.cpp]
!8 = metadata !{metadata !"0x1f\00\000\000\000\000\000", null, null, metadata !9, metadata !"_ZTS3Foo"} ; [ DW_TAG_ptr_to_member_type ] [line 0, size 0, align 0, offset 0] [from int]
!9 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!11 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
