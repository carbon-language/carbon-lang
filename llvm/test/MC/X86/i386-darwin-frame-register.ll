; RUN: llc -filetype=obj %s -o - | llvm-dwarfdump -debug-dump=frames - | FileCheck %s

; IR reduced from a dummy:
; void foo() {}

; x86 Darwin uses different register mappings for eh_frame and debug_frame
; sections. Check that the right mapping is used in debug_frame.
; In the debug_frame mapping, regsiter 4 is ESP, thus the below tests that
; the CFA is ESP+4 upon function entry.

; CHECK: .debug_frame contents:
; CHECK: ffffffff CIE
; CHECK-NOT: {{CIE|FDE}}
; CHECK:   DW_CFA_def_cfa: reg4 +4

; ModuleID = 'foo.c'
target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.10.0"

; Function Attrs: nounwind ssp
define void @foo() #0 {
entry:
  ret void
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = !{!"0x11\0012\00clang version 3.7.0 (trunk 230514) (llvm/trunk 230518)\000\00\000\00\001", !1, !2, !2, !2, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp/foo.c] [DW_LANG_C99]
!1 = !{!"foo.c", !"/tmp"}
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 2}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 3.7.0 (trunk 230514) (llvm/trunk 230518)"}
