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

!0 = !{!"0x11\0012\00clang version 3.5 \000\00\000\00\000", !1, !2, !2, !2, !3, !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/simple.c] [DW_LANG_C99]
!1 = !{!"simple.c", !"/tmp/dbginfo"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x34\00f\00f\00\002\000\001", null, !5, !6, %struct.foo* @f, null} ; [ DW_TAG_variable ] [f] [line 2] [def]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/simple.c]
!6 = !{!"0x13\00foo\001\000\008\000\000\000", !1, null, null, !2, null, null, null} ; [ DW_TAG_structure_type ] [foo] [line 1, size 0, align 8, offset 0] [def] [from ]
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 1, !"Debug Info Version", i32 2}
!9 = !{!"clang version 3.5 "}
