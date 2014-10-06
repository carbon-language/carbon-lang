; RUN: llc -O0 -filetype=obj -mtriple=arm-none-linux < %s | llvm-dwarfdump - | FileCheck %s

; CHECK: file format ELF32-arm-little

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.6.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !2, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/a/empty.c] [DW_LANG_C99]
!1 = metadata !{metadata !"empty.c", metadata !"/a"}
!2 = metadata !{}
!3 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!4 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!5 = metadata !{i32 1, metadata !"wchar_size", i32 4}
!6 = metadata !{i32 1, metadata !"min_enum_size", i32 4}
!7 = metadata !{metadata !"clang version 3.6.0 "}
