; RUN: llc < %s | FileCheck -check-prefix=CHECK-HEADER %s
; RUN: llc < %s | FileCheck -check-prefix=CHECK-CODE %s
; RUN: llc < %s | FileCheck -check-prefix=CHECK-DATA %s
; RUN: llc < %s | FileCheck -check-prefix=CHECK-BSS %s
; RUN: llc < %s | FileCheck -check-prefix=CHECK-CUSTOM %s


; -- header --
; CHECK-HEADER: .short 2 # DWARF Arange version number
; CHECK-HEADER-NEXT: .long .L.debug_info_begin0
; CHECK-HEADER-NEXT: .byte 8 # Address Size (in bytes)
; CHECK-HEADER-NEXT: .byte 0 # Segment Size (in bytes)
; -- alignment --
; CHECK-HEADER-NEXT: .byte
; CHECK-HEADER-NEXT: .byte
; CHECK-HEADER-NEXT: .byte
; CHECK-HEADER-NEXT: .byte
; -- finish --
; CHECK-HEADER: # ARange terminator

; <text section> - it should have made one span covering all functions in this CU.
; CHECK-CODE: .short 2 # DWARF Arange version number
; CHECK-CODE: .quad .Lfunc_begin0
; CHECK-CODE-NEXT: .Lset1 = .L.text_end-.Lfunc_begin0
; CHECK-CODE: # ARange terminator

; <data section> - it should have made one span covering all vars in this CU.
; CHECK-DATA: .short 2 # DWARF Arange version number
; CHECK-DATA: .quad some_data
; CHECK-DATA-NEXT: -some_data
; CHECK-DATA: # ARange terminator

; <common symbols> - it should have made one span for each symbol.
; CHECK-BSS: .short 2 # DWARF Arange version number
; CHECK-BSS: .quad some_bss
; CHECK-BSS-NEXT: .quad 4
; CHECK-BSS: # ARange terminator

; <other sections> - it should have made one span covering all vars in this CU.
; CHECK-CUSTOM: .short 2 # DWARF Arange version number
; CHECK-CUSTOM: .quad some_other
; CHECK-CUSTOM-NEXT: -some_other
; CHECK-CUSTOM: # ARange terminator




; -- source code --
; Generated from: "clang -c -g -emit-llvm"
;
; int some_data = 4;
; int some_bss;
; int some_other __attribute__ ((section ("strangesection"))) = 5;
; 
; void some_code()
; {
;    some_bss += some_data + some_other;
; }

target triple = "x86_64-unknown-linux-gnu"

@some_data = global i32 4, align 4
@some_other = global i32 5, section "strangesection", align 4
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
!llvm.module.flags = !{!13}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.4 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !8, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/home/kayamon/test.c] [DW_LANG_C99]
!1 = metadata !{metadata !"test.c", metadata !"/home/kayamon"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"some_code", metadata !"some_code", metadata !"", i32 5, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, void ()* @some_code, null, null, metadata !2, i32 6} ; [ DW_TAG_subprogram ] [line 5] [def] [scope 6] [some_code]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/home/kayamon/test.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{metadata !9, metadata !11, metadata !12}
!9 = metadata !{i32 786484, i32 0, null, metadata !"some_data", metadata !"some_data", metadata !"", metadata !5, i32 1, metadata !10, i32 0, i32 1, i32* @some_data, null} ; [ DW_TAG_variable ] [some_data] [line 1] [def]
!10 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!11 = metadata !{i32 786484, i32 0, null, metadata !"some_other", metadata !"some_other", metadata !"", metadata !5, i32 3, metadata !10, i32 0, i32 1, i32* @some_other, null} ; [ DW_TAG_variable ] [some_other] [line 3] [def]
!12 = metadata !{i32 786484, i32 0, null, metadata !"some_bss", metadata !"some_bss", metadata !"", metadata !5, i32 2, metadata !10, i32 0, i32 1, i32* @some_bss, null} ; [ DW_TAG_variable ] [some_bss] [line 2] [def]
!13 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!14 = metadata !{i32 7, i32 0, metadata !4, null}
!15 = metadata !{i32 8, i32 0, metadata !4, null} ; [ DW_TAG_imported_declaration ]
