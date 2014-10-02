; REQUIRES: object-emission

; RUN: llc -o - %s -filetype=obj -O0 -generate-dwarf-pub-sections=Disable -generate-type-units -mtriple=x86_64-unknown-linux-gnu | llvm-dwarfdump -debug-dump=types - | FileCheck %s

; struct foo {
; } f;

; no known LLVM frontends produce appropriate unique identifiers for C types,
; so we don't produce type units for them
; CHECK-NOT: DW_TAG_type_unit

%struct.foo = type {}

@f = common global %struct.foo zeroinitializer, align 1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.5 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !2, metadata !3, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/simple.c] [DW_LANG_C99]
!1 = metadata !{metadata !"simple.c", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786484, i32 0, null, metadata !"f", metadata !"f", metadata !"", metadata !5, i32 2, metadata !6, i32 0, i32 1, %struct.foo* @f, null} ; [ DW_TAG_variable ] [f] [line 2] [def]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/simple.c]
!6 = metadata !{i32 786451, metadata !1, null, metadata !"foo", i32 1, i64 0, i64 8, i32 0, i32 0, null, metadata !2, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [foo] [line 1, size 0, align 8, offset 0] [def] [from ]
!7 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!8 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!9 = metadata !{metadata !"clang version 3.5 "}
