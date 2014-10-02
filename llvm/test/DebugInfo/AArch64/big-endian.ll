; RUN: llc %s -filetype=asm -o -

target datalayout = "E-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64_be--none-eabi"

@a = common global i32 0, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.6.0 ", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !2, metadata !3, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/work/validation/-] [DW_LANG_C99]
!1 = metadata !{metadata !"-", metadata !"/work/validation"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786484, i32 0, null, metadata !"a", metadata !"a", metadata !"", metadata !5, i32 1, metadata !7, i32 0, i32 1, i32* @a, null} ; [ DW_TAG_variable ] [a] [line 1] [def]
!5 = metadata !{i32 786473, metadata !6}          ; [ DW_TAG_file_type ] [/work/validation/<stdin>]
!6 = metadata !{metadata !"<stdin>", metadata !"/work/validation"}
!7 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!8 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!9 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!10 = metadata !{metadata !"clang version 3.6.0 "}
