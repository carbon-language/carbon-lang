; RUN: llc  -filetype=asm %s -o - | FileCheck %s

; Verifies that the DWARF* sections come _after_ the __TEXT sections.
; rdar://problem/15623193

; CHECK: .section	__TEXT,__text,
; CHECK-NOT: __DWARF,__debug
; CHECK: .section	__TEXT,__cstring,cstring_literals
target triple = "thumbv7-apple-ios"

!llvm.module.flags = !{!3, !4}
!llvm.dbg.cu = !{!0}

!0 = !MDCompileUnit(language: DW_LANG_C99, producer: "LLVM", isOptimized: true, file: !5, enums: !1, retainedTypes: !1, subprograms: !1, globals: !1)
!1 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 2}
!4 = !{i32 1, !"Debug Info Version", i32 3}
!5 = !MDFile(filename: "test.c", directory: "/Volumes/Data/radar/15623193")
