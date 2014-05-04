; RUN: llc -mtriple thumbv7--windows-itanium -filetype obj -o - %s \
; RUN:     | llvm-readobj -r - | FileCheck %s

; ModuleID = '/Users/compnerd/work/llvm/test/MC/ARM/reduced.c'
target datalayout = "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7--windows-itanium"

define arm_aapcs_vfpcc void @function() {
entry:
  ret void, !dbg !0
}

!llvm.dbg.cu = !{!7}
!llvm.module.flags = !{!9, !10}

!0 = metadata !{i32 1, i32 0, metadata !1, null}
!1 = metadata !{i32 786478, metadata !2, metadata !3, metadata !"function", metadata !"function", metadata !"", i32 1, metadata !4, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, void ()* @function, null, null, metadata !6, i32 1} ; [ DW_TAG_subprogram ], [line 1], [def], [function]
!2 = metadata !{metadata !"/Users/compnerd/work/llvm/test/MC/ARM/reduced.c", metadata !"/Users/compnerd/work/llvm"}
!3 = metadata !{i32 786473, metadata !2} ; [ DW_TAG_file_type] [/Users/compnerd/work/llvm/test/MC/ARM/reduced.c]
!4 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !5, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ], [line 0, size 0, align 0, offset 0] [from ]
!5 = metadata !{null}
!6 = metadata !{}
!7 = metadata !{i32 786449, metadata !2, i32 12, metadata !"clang version 3.5.0", i1 false, metadata !"", i32 0, metadata !6, metadata !6, metadata !8, metadata !6, metadata !6, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/Users/compnerd/work/llvm/test/MC/ARM/reduced.c] [DW_LANG_C99]
!8 = metadata !{metadata !1}
!9 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!10 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}

; CHECK: Relocations [
; CHECK:   Section {{.*}} .debug_info {
; CHECK:     0x6 IMAGE_REL_ARM_SECREL .debug_abbrev
; CHECK:     0xC IMAGE_REL_ARM_SECREL .debug_str
; CHECK:     0x12 IMAGE_REL_ARM_SECREL .debug_str
; CHECK:     0x16 IMAGE_REL_ARM_SECREL .debug_line
; CHECK:     0x1A IMAGE_REL_ARM_SECREL .debug_str
; CHECK:     0x27 IMAGE_REL_ARM_SECREL .debug_str
; CHECK:   }
; CHECK:   Section {{.*}}.debug_pubnames {
; CHECK:     0x6 IMAGE_REL_ARM_SECREL .debug_info
; CHECK:   }
; CHECK: ]

