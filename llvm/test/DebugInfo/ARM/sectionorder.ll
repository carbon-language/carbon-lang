; RUN: llc  -filetype=asm %s -o - | FileCheck %s

; Verifies that the DWARF* sections come _after_ the __TEXT sections.
; rdar://problem/15623193

; CHECK: .section	__TEXT,__text,
; CHECK-NOT: __DWARF,__debug
; CHECK: .section	__TEXT,__cstring,cstring_literals
target triple = "thumbv7-apple-ios"

!llvm.module.flags = !{!3, !4}
!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 786449, i32 0, i32 12, metadata !"test.c", metadata !"/Volumes/Data/radar/15623193", metadata !"LLVM", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !1, metadata !1} ; [ DW_TAG_compile_unit ] [/Volumes/Data/radar/15623193/test.c] [DW_LANG_C99]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!4 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
