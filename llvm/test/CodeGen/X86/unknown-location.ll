; RUN: llc < %s -asm-verbose=false -mtriple=x86_64-apple-darwin10 -use-unknown-locations | FileCheck %s

; The divide instruction does not have a debug location. CodeGen should
; represent this in the debug information. This is done by setting line
; and column to 0

;      CHECK:         leal
; CHECK-NEXT:         .loc 1 0 0
;      CHECK:         cltd
; CHECK-NEXT:         idivl
; CHECK-NEXT:         .loc 2 4 3

define i32 @foo(i32 %w, i32 %x, i32 %y, i32 %z) nounwind {
entry:
  %a = add  i32 %w, %x, !dbg !8
  %b = sdiv i32 %a, %y
  %c = add  i32 %b, %z, !dbg !8
  ret i32 %c, !dbg !8
}

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!12}

!0 = metadata !{metadata !"0x101\00x\001\000", metadata !1, metadata !2, metadata !6} ; [ DW_TAG_arg_variable ]
!1 = metadata !{metadata !"0x2e\00foo\00foo\00foo\001\000\001\000\006\000\000\001", metadata !10, metadata !2, metadata !4, null, i32 (i32, i32, i32, i32)* @foo, null, null, null} ; [ DW_TAG_subprogram ]
!2 = metadata !{metadata !"0x29", metadata !10} ; [ DW_TAG_file_type ]
!3 = metadata !{metadata !"0x11\0012\00producer\000\00\000\00\000", metadata !10, metadata !11, metadata !11, metadata !9, null, null} ; [ DW_TAG_compile_unit ]
!4 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !10, metadata !2, null, metadata !5, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = metadata !{metadata !6}
!6 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !10, metadata !2} ; [ DW_TAG_base_type ]
!7 = metadata !{metadata !"0xb\001\0030\000", metadata !2, metadata !1} ; [ DW_TAG_lexical_block ]
!8 = metadata !{i32 4, i32 3, metadata !7, null}
!9 = metadata !{metadata !1}
!10 = metadata !{metadata !"test.c", metadata !"/dir"}
!11 = metadata !{i32 0}
!12 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
