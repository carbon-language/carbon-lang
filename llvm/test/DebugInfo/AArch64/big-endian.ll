; RUN: llc %s -filetype=asm -o -

target datalayout = "E-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64_be--none-eabi"

@a = common global i32 0, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = !{!"0x11\0012\00clang version 3.6.0 \001\00\000\00\001", !1, !2, !2, !2, !3, !2} ; [ DW_TAG_compile_unit ] [/work/validation/-] [DW_LANG_C99]
!1 = !{!"-", !"/work/validation"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x34\00a\00a\00\001\000\001", null, !5, !7, i32* @a, null} ; [ DW_TAG_variable ] [a] [line 1] [def]
!5 = !{!"0x29", !6}          ; [ DW_TAG_file_type ] [/work/validation/<stdin>]
!6 = !{!"<stdin>", !"/work/validation"}
!7 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 2}
!10 = !{!"clang version 3.6.0 "}
