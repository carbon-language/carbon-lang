; RUN: llc  -filetype=asm %s -o - | FileCheck %s

; Verifies that the DWARF* sections come _after_ the __TEXT sections.
; rdar://problem/15623193

; CHECK: .section	__TEXT,__text,
; CHECK-NOT: __DWARF,__debug
; CHECK: .section	__TEXT,__cstring,cstring_literals
target triple = "thumbv7-apple-ios"

!llvm.module.flags = !{!3, !4}
!llvm.dbg.cu = !{!0}

!0 = !{!"0x11\0012\00LLVM\001\00\00\00\00", !5, !1, !1, !1, !1, null} ; [ DW_TAG_compile_unit ] [/Volumes/Data/radar/15623193/test.c] [DW_LANG_C99]
!1 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 2}
!4 = !{i32 1, !"Debug Info Version", i32 2}
!5 = !{!"test.c", !"/Volumes/Data/radar/15623193"}
