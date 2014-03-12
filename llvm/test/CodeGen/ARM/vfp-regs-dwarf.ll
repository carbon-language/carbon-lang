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

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.5.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/Users/tim/llvm/build/tmp.c] [DW_LANG_C99]
!1 = metadata !{metadata !"tmp.c", metadata !"/Users/tim/llvm/build"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"bar", metadata !"bar", metadata !"", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, void ()* @stack_offsets, null, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [bar]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/Users/tim/llvm/build/tmp.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!9 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}

