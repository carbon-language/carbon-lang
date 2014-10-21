; RUN: llc -generate-arange-section < %s | FileCheck %s

; First CU
; CHECK:      .long   44                      # Length of ARange Set
; CHECK-NEXT: .short  2                       # DWARF Arange version number
; CHECK-NEXT: .long   .L.debug_info_begin0    # Offset Into Debug Info Section
; CHECK-NEXT: .byte   8                       # Address Size (in bytes)
; CHECK-NEXT: .byte   0                       # Segment Size (in bytes)
; CHECK-NEXT: .zero   4,255
; CHECK-NEXT: .quad   kittens
; CHECK-NEXT: .quad   rainbows-kittens
; CHECK-NEXT: .quad   0                       # ARange terminator
; CHECK-NEXT: .quad   0

; Second CU
; CHECK-NEXT: .long   44                      # Length of ARange Set
; CHECK-NEXT: .short  2                       # DWARF Arange version number
; CHECK-NEXT: .long   .L.debug_info_begin1    # Offset Into Debug Info Section
; CHECK-NEXT: .byte   8                       # Address Size (in bytes)
; CHECK-NEXT: .byte   0                       # Segment Size (in bytes)
; CHECK-NEXT: .zero   4,255
; CHECK-NEXT: .quad   rainbows
; CHECK-NEXT: .quad   .Ldebug_end0-rainbows
; CHECK-NEXT: .quad   0                       # ARange terminator
; CHECK-NEXT: .quad   0


; Generated from: clang -c -g -emit-llvm
;                 llvm-link test1.bc test2.bc -o test.bc
; test1.c: int kittens = 4;
; test2.c: int rainbows = 5;




; ModuleID = 'test.bc'
target triple = "x86_64-unknown-linux-gnu"

@kittens = global i32 4, align 4
@rainbows = global i32 5, align 4

!llvm.dbg.cu = !{!0, !7}
!llvm.module.flags = !{!12, !13}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.4 \000\00\000\00\000", metadata !1, metadata !2, metadata !2, metadata !2, metadata !3, metadata !2} ; [ DW_TAG_compile_unit ] [/home/kayamon/test1.c] [DW_LANG_C99]
!1 = metadata !{metadata !"test1.c", metadata !"/home/kayamon"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x34\00kittens\00kittens\00\001\000\001", null, metadata !5, metadata !6, i32* @kittens, null} ; [ DW_TAG_variable ] [kittens] [line 1] [def]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/home/kayamon/test1.c]
!6 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!7 = metadata !{metadata !"0x11\0012\00clang version 3.4 \000\00\000\00\000", metadata !8, metadata !2, metadata !2, metadata !2, metadata !9, metadata !2} ; [ DW_TAG_compile_unit ] [/home/kayamon/test2.c] [DW_LANG_C99]
!8 = metadata !{metadata !"test2.c", metadata !"/home/kayamon"}
!9 = metadata !{metadata !10}
!10 = metadata !{metadata !"0x34\00rainbows\00rainbows\00\001\000\001", null, metadata !11, metadata !6, i32* @rainbows, null} ; [ DW_TAG_variable ] [rainbows] [line 1] [def]
!11 = metadata !{metadata !"0x29", metadata !8}         ; [ DW_TAG_file_type ] [/home/kayamon/test2.c]
!12 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!13 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
