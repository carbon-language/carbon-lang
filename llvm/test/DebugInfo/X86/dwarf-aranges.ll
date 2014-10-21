; RUN: llc -generate-arange-section < %s | FileCheck %s


; -- header --
; CHECK: .short 2 # DWARF Arange version number
; CHECK-NEXT: .long .L.debug_info_begin0
; CHECK-NEXT: .byte 8 # Address Size (in bytes)
; CHECK-NEXT: .byte 0 # Segment Size (in bytes)
; -- alignment --
; CHECK-NEXT: .zero 4,255

; <common symbols> - it should have made one span for each symbol.
; CHECK-NEXT: .quad some_bss
; CHECK-NEXT: .quad 4

; <data section> - it should have made one span covering all vars in this CU.
; CHECK-NEXT: .quad some_data
; CHECK-NEXT: .quad .Ldebug_end1-some_data

; <text section> - it should have made one span covering all functions in this CU.
; CHECK-NEXT: .quad .Lfunc_begin0
; CHECK-NEXT: .quad .Ldebug_end2-.Lfunc_begin0

; <other sections> - it should have made one span covering all vars in this CU.
; CHECK-NEXT: .quad some_other
; CHECK-NEXT: .quad .Ldebug_end3-some_other

; -- finish --
; CHECK-NEXT: # ARange terminator



; -- source code --
; Generated from: "clang -c -g -emit-llvm"
;
; int some_data = 4;
; int some_bss;
; int some_other __attribute__ ((section ("strange+section"))) = 5;
; 
; void some_code()
; {
;    some_bss += some_data + some_other;
; }

target triple = "x86_64-unknown-linux-gnu"

@some_data = global i32 4, align 4
@some_other = global i32 5, section "strange+section", align 4
@some_bss = common global i32 0, align 4

define void @some_code() {
entry:
  %0 = load i32* @some_data, align 4, !dbg !14
  %1 = load i32* @some_other, align 4, !dbg !14
  %add = add nsw i32 %0, %1, !dbg !14
  %2 = load i32* @some_bss, align 4, !dbg !14
  %add1 = add nsw i32 %2, %add, !dbg !14
  store i32 %add1, i32* @some_bss, align 4, !dbg !14
  ret void, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !16}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.4 \000\00\000\00\000", metadata !1, metadata !2, metadata !2, metadata !3, metadata !8, metadata !2} ; [ DW_TAG_compile_unit ] [/home/kayamon/test.c] [DW_LANG_C99]
!1 = metadata !{metadata !"test.c", metadata !"/home/kayamon"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00some_code\00some_code\00\005\000\001\000\006\000\000\006", metadata !1, metadata !5, metadata !6, null, void ()* @some_code, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 5] [def] [scope 6] [some_code]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/home/kayamon/test.c]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{metadata !9, metadata !11, metadata !12}
!9 = metadata !{metadata !"0x34\00some_data\00some_data\00\001\000\001", null, metadata !5, metadata !10, i32* @some_data, null} ; [ DW_TAG_variable ] [some_data] [line 1] [def]
!10 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!11 = metadata !{metadata !"0x34\00some_other\00some_other\00\003\000\001", null, metadata !5, metadata !10, i32* @some_other, null} ; [ DW_TAG_variable ] [some_other] [line 3] [def]
!12 = metadata !{metadata !"0x34\00some_bss\00some_bss\00\002\000\001", null, metadata !5, metadata !10, i32* @some_bss, null} ; [ DW_TAG_variable ] [some_bss] [line 2] [def]
!13 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!14 = metadata !{i32 7, i32 0, metadata !4, null}
!15 = metadata !{i32 8, i32 0, metadata !4, null}
!16 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
