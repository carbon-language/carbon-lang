; RUN: llc -filetype=obj -O0 < %s -mtriple x86_64-none-linux | \
; RUN:     llvm-dwarfdump - 2>&1 | FileCheck %s
; RUN: llc -filetype=obj -O0 < %s -mtriple i386-none-linux | \
; RUN:     llvm-dwarfdump - 2>&1 | FileCheck %s
; RUN: llc -filetype=obj -O0 < %s -mtriple x86_64-none-mingw32 | \
; RUN:     llvm-dwarfdump - 2>&1 | FileCheck %s
; RUN: llc -filetype=obj -O0 < %s -mtriple i386-none-mingw32 | \
; RUN:     llvm-dwarfdump - 2>&1 | FileCheck %s

; CHECK-NOT: failed to compute relocation

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.6.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !2, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/a/empty.c] [DW_LANG_C99]
!1 = metadata !{metadata !"empty.c", metadata !"/a"}
!2 = metadata !{}
!3 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!4 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!5 = metadata !{metadata !"clang version 3.6.0 "}
