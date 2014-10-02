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

!0 = metadata !{metadata !"0x11\0012\00clang version 3.5 \000\00\000\00\000", metadata !1, metadata !2, metadata !2, metadata !2, metadata !3, metadata !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/simple.c] [DW_LANG_C99]
!1 = metadata !{metadata !"simple.c", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x34\00f\00f\00\002\000\001", null, metadata !5, metadata !6, %struct.foo* @f, null} ; [ DW_TAG_variable ] [f] [line 2] [def]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/simple.c]
!6 = metadata !{metadata !"0x13\00foo\001\000\008\000\000\000", metadata !1, null, null, metadata !2, null, null, null} ; [ DW_TAG_structure_type ] [foo] [line 1, size 0, align 8, offset 0] [def] [from ]
!7 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!8 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!9 = metadata !{metadata !"clang version 3.5 "}
