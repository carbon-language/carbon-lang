; RUN: llc < %s -mtriple=arm-linux-unknown-gnueabi -verify-machineinstrs -filetype=asm | FileCheck %s -check-prefix=ARM-linux
; RUN: llc < %s -mtriple=arm-linux-unknown-gnueabi -filetype=obj

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

define void @test_basic() #0 {
        %mem = alloca i32, i32 10
        call void @dummy_use (i32* %mem, i32 10)
	ret void

; ARM-linux:      test_basic:

; ARM-linux:      push    {r4, r5}
; ARM-linux:      .cfi_def_cfa_offset 8
; ARM-linux:      .cfi_offset r5, -4
; ARM-linux:      .cfi_offset r4, -8
; ARM-linux-NEXT: mrc     p15, #0, r4, c13, c0, #3
; ARM-linux-NEXT: mov     r5, sp
; ARM-linux-NEXT: ldr     r4, [r4, #4]
; ARM-linux-NEXT: cmp     r4, r5
; ARM-linux-NEXT: blo     .LBB0_2

; ARM-linux:      mov     r4, #48
; ARM-linux-NEXT: mov     r5, #0
; ARM-linux-NEXT: stmdb   sp!, {lr}
; ARM-linux:      .cfi_def_cfa_offset 12
; ARM-linux:      .cfi_offset lr, -12
; ARM-linux-NEXT: bl      __morestack
; ARM-linux-NEXT: ldm     sp!, {lr}
; ARM-linux-NEXT: pop     {r4, r5}
; ARM-linux:      .cfi_def_cfa_offset 0
; ARM-linux-NEXT: bx      lr

; ARM-linux:      pop     {r4, r5}
; ARM-linux:      .cfi_def_cfa_offset 0
; ARM-linux       .cfi_same_value r4
; ARM-linux       .cfi_same_value r5
}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.5 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/tmp/var.c] [DW_LANG_C99]
!1 = metadata !{metadata !"var.c", metadata !"/tmp"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"test_basic",
  metadata !"test_basic", metadata !"", i32 5, metadata !6, i1 false, i1 true,
  i32 0, i32 0, null, i32 256, i1 false, void ()* @test_basic, null, null, metadata !2, i32 5} ; [ DW_TAG_subprogram ] [line 5] [def] [sum]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/tmp/var.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !8}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!10 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!11 = metadata !{metadata !"clang version 3.5 "}
!12 = metadata !{i32 786689, metadata !4, metadata !"count", metadata !5, i32 16777221, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [count] [line 5]
!13 = metadata !{i32 5, i32 0, metadata !4, null}
!14 = metadata !{i32 786688, metadata !4, metadata !"vl", metadata !5, i32 6, metadata !15, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [vl] [line 6]
!15 = metadata !{i32 786454, metadata !16, null, metadata !"va_list", i32 30, i64 0, i64 0, i64 0, i32 0, metadata !17} ; [ DW_TAG_typedef ] [va_list] [line 30, size 0, align 0, offset 0] [from __builtin_va_list]
!16 = metadata !{metadata !"/linux-x86_64-high/gcc_4.7.2/dbg/llvm/bin/../lib/clang/3.5/include/stdarg.h", metadata !"/tmp"}
!17 = metadata !{i32 786454, metadata !1, null, metadata !"__builtin_va_list", i32 6, i64 0, i64 0, i64 0, i32 0, metadata !18} ; [ DW_TAG_typedef ] [__builtin_va_list] [line 6, size 0, align 0, offset 0] [from __va_list]
!18 = metadata !{i32 786451, metadata !1, null, metadata !"__va_list", i32 6, i64 32, i64 32, i32 0, i32 0, null, metadata !19, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [__va_list] [line 6, size 32, align 32, offset 0] [def] [from ]
!19 = metadata !{metadata !20}
!20 = metadata !{i32 786445, metadata !1, metadata !18, metadata !"__ap", i32 6, i64 32, i64 32, i64 0, i32 0, metadata !21} ; [ DW_TAG_member ] [__ap] [line 6, size 32, align 32, offset 0] [from ]
!21 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, null} ; [ DW_TAG_pointer_type ] [line 0, size 32, align 32, offset 0] [from ]
!22 = metadata !{i32 6, i32 0, metadata !4, null}
!23 = metadata !{i32 7, i32 0, metadata !4, null}
!24 = metadata !{i32 786688, metadata !4, metadata !"test_basic", metadata !5, i32 8, metadata !8, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [sum] [line 8]
!25 = metadata !{i32 8, i32 0, metadata !4, null} ; [ DW_TAG_imported_declaration ]
!26 = metadata !{i32 786688, metadata !27, metadata !"i", metadata !5, i32 9, metadata !8, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [i] [line 9]
!27 = metadata !{i32 786443, metadata !1, metadata !4, i32 9, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/tmp/var.c]
!28 = metadata !{i32 9, i32 0, metadata !27, null}
!29 = metadata !{i32 10, i32 0, metadata !30, null}
!30 = metadata !{i32 786443, metadata !1, metadata !27, i32 9, i32 0, i32 1} ; [ DW_TAG_lexical_block ] [/tmp/var.c]
!31 = metadata !{i32 11, i32 0, metadata !30, null}
!32 = metadata !{i32 12, i32 0, metadata !4, null}
!33 = metadata !{i32 13, i32 0, metadata !4, null}

; Just to prevent the alloca from being optimized away
declare void @dummy_use(i32*, i32)

attributes #0 = { "split-stack" }
