; RUN: llc  -filetype=asm %s -o - | FileCheck %s

; Verifies that the DWARF* sections come _after_ the __TEXT sections.
; rdar://problem/15623193

; CHECK: .section	__TEXT,__text,
; CHECK-NOT: __DWARF,__debug
; CHECK: .section	__TEXT,__cstring,cstring_literals
target triple = "thumbv7-apple-ios"

!llvm.module.flags = !{!3, !4}
!llvm.dbg.cu = !{!0}

!0 = metadata !{metadata !"0x11\0012\00LLVM\001\00\00\00\00", metadata !5, metadata !1, metadata !1, metadata !1, metadata !1, null} ; [ DW_TAG_compile_unit ] [/Volumes/Data/radar/15623193/test.c] [DW_LANG_C99]
!1 = metadata !{}
!3 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!4 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!5 = metadata !{metadata !"test.c", metadata !"/Volumes/Data/radar/15623193"}
