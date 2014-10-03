; RUN: llc -mtriple=armv7-linux-gnueabihf %s -o - | FileCheck %s

; Generated from:
;     void stack_offsets() {
;       asm("" ::: "d8", "d9", "d11", "d13");
;     }
; Compiled with: "clang -target armv7-linux-gnueabihf -O3"

; The important point we're checking here is that the .cfi directives describe
; the layout of the VFP registers correctly. The fact that the numbers are
; monotonic in memory is also a nice property to have.

define void @stack_offsets() {
; CHECK-LABEL: stack_offsets:
; CHECK: vpush {d13}
; CHECK: vpush {d11}
; CHECK: vpush {d8, d9}

; CHECK: .cfi_offset {{269|d13}}, -8
; CHECK: .cfi_offset {{267|d11}}, -16
; CHECK: .cfi_offset {{265|d9}}, -24
; CHECK: .cfi_offset {{264|d8}}, -32

; CHECK: vpop {d8, d9}
; CHECK: vpop {d11}
; CHECK: vpop {d13}
  call void asm sideeffect "", "~{d8},~{d9},~{d11},~{d13}"() #1
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.5.0 \000\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [/Users/tim/llvm/build/tmp.c] [DW_LANG_C99]
!1 = metadata !{metadata !"tmp.c", metadata !"/Users/tim/llvm/build"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00bar\00bar\00\001\000\001\000\006\000\000\001", metadata !1, metadata !5, metadata !6, null, void ()* @stack_offsets, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 1] [def] [bar]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/Users/tim/llvm/build/tmp.c]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!9 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}

