; RUN: llc -filetype=asm -o - < %s -mtriple arm-arm-none-eabi -disable-fp-elim| FileCheck %s --check-prefix=CHECK-ARM
; RUN: llc -filetype=asm -o - < %s -mtriple arm-arm-none-eabi | FileCheck %s --check-prefix=CHECK-ARM-FP-ELIM

define void @test1() {
    %tmp = alloca [ 64 x i32 ] , align 4
    ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.5 \000\00\000\00\000", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [/tmp/large.c] [DW_LANG_C99]
!1 = metadata !{metadata !"large.c", metadata !"/tmp"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00test1\00test1\00\001\000\001\000\006\000\000\001", metadata !1, metadata !5, metadata !6, null, void ()* @test1, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 1] [def] [test1]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/tmp/large.c]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!9 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!10 = metadata !{metadata !"clang version 3.5 "}
!11 = metadata !{i32 2, i32 0, metadata !4, null}

; CHECK-ARM-LABEL: test1:
; CHECK-ARM: .cfi_startproc
; CHECK-ARM: sub    sp, sp, #256
; CHECK-ARM: .cfi_endproc

; CHECK-ARM-FP-ELIM-LABEL: test1:
; CHECK-ARM-FP-ELIM: .cfi_startproc
; CHECK-ARM-FP-ELIM: sub    sp, sp, #256
; CHECK-ARM-FP-ELIM: .cfi_endproc

define void @test2() {
    %tmp = alloca [ 4168 x i8 ] , align 4
    ret void
}

; CHECK-ARM-LABEL: test2:
; CHECK-ARM: .cfi_startproc
; CHECK-ARM: push    {r4, r5}
; CHECK-ARM: .cfi_def_cfa_offset 8
; CHECK-ARM: .cfi_offset r5, -4
; CHECK-ARM: .cfi_offset r4, -8
; CHECK-ARM: sub    sp, sp, #72
; CHECK-ARM: sub    sp, sp, #4096
; CHECK-ARM: .cfi_def_cfa_offset 4176
; CHECK-ARM: .cfi_endproc

; CHECK-ARM-FP_ELIM-LABEL: test2:
; CHECK-ARM-FP_ELIM: .cfi_startproc
; CHECK-ARM-FP_ELIM: push    {r4, r5}
; CHECK-ARM-FP_ELIM: .cfi_def_cfa_offset 8
; CHECK-ARM-FP_ELIM: .cfi_offset 54, -4
; CHECK-ARM-FP_ELIM: .cfi_offset r4, -8
; CHECK-ARM-FP_ELIM: sub    sp, sp, #72
; CHECK-ARM-FP_ELIM: sub    sp, sp, #4096
; CHECK-ARM-FP_ELIM: .cfi_def_cfa_offset 4176
; CHECK-ARM-FP_ELIM: .cfi_endproc

define i32 @test3() {
	%retval = alloca i32, align 4
	%tmp = alloca i32, align 4
	%a = alloca [805306369 x i8], align 16
	store i32 0, i32* %tmp
	%tmp1 = load i32* %tmp
        ret i32 %tmp1
}

; CHECK-ARM-LABEL: test3:
; CHECK-ARM: .cfi_startproc
; CHECK-ARM: push    {r4, r5, r11}
; CHECK-ARM: .cfi_def_cfa_offset 12
; CHECK-ARM: .cfi_offset r11, -4
; CHECK-ARM: .cfi_offset r5, -8
; CHECK-ARM: .cfi_offset r4, -12
; CHECK-ARM: add    r11, sp, #8
; CHECK-ARM: .cfi_def_cfa r11, 4
; CHECK-ARM: sub    sp, sp, #20
; CHECK-ARM: sub    sp, sp, #805306368
; CHECK-ARM: bic    sp, sp, #15
; CHECK-ARM: .cfi_endproc

; CHECK-ARM-FP-ELIM-LABEL: test3:
; CHECK-ARM-FP-ELIM: .cfi_startproc
; CHECK-ARM-FP-ELIM: push    {r4, r5, r11}
; CHECK-ARM-FP-ELIM: .cfi_def_cfa_offset 12
; CHECK-ARM-FP-ELIM: .cfi_offset r11, -4
; CHECK-ARM-FP-ELIM: .cfi_offset r5, -8
; CHECK-ARM-FP-ELIM: .cfi_offset r4, -12
; CHECK-ARM-FP-ELIM: add    r11, sp, #8
; CHECK-ARM-FP-ELIM: .cfi_def_cfa r11, 4
; CHECK-ARM-FP-ELIM: sub    sp, sp, #20
; CHECK-ARM-FP-ELIM: sub    sp, sp, #805306368
; CHECK-ARM-FP-ELIM: bic    sp, sp, #15
; CHECK-ARM-FP-ELIM: .cfi_endproc

