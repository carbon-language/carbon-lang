; RUN: llc -O0 -filetype=obj -mtriple=aarch64-none-linux < %s | llvm-dwarfdump - | FileCheck %s

; CHECK: file format ELF64-aarch64-little

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.6.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !2, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/a/empty.c] [DW_LANG_C99]
!1 = metadata !{metadata !"empty.c", metadata !"/a"}
!2 = metadata !{}
!3 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!4 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!5 = metadata !{metadata !"clang version 3.6.0 "}
