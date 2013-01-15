; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; Check that module flags are structurally correct.
;
; CHECK: incorrect number of operands in module flag
; CHECK: metadata !0
!0 = metadata !{ i32 1 }
; CHECK: invalid behavior operand in module flag (expected constant integer)
; CHECK: metadata !"foo"
!1 = metadata !{ metadata !"foo", metadata !"foo", i32 42 }
; CHECK: invalid behavior operand in module flag (unexpected constant)
; CHECK: i32 999
!2 = metadata !{ i32 999, metadata !"foo", i32 43 }
; CHECK: invalid ID operand in module flag (expected metadata string)
; CHECK: i32 1
!3 = metadata !{ i32 1, i32 1, i32 44 }
; CHECK: invalid value for 'require' module flag (expected metadata pair)
; CHECK: i32 45
!4 = metadata !{ i32 3, metadata !"bla", i32 45 }
; CHECK: invalid value for 'require' module flag (expected metadata pair)
; CHECK: metadata !
!5 = metadata !{ i32 3, metadata !"bla", metadata !{ i32 46 } }
; CHECK: invalid value for 'require' module flag (first value operand should be a string)
; CHECK: i32 47
!6 = metadata !{ i32 3, metadata !"bla", metadata !{ i32 47, i32 48 } }

; Check that module flags only have unique IDs.
;
; CHECK: module flag identifiers must be unique (or of 'require' type)
!7 = metadata !{ i32 1, metadata !"foo", i32 49 }
!8 = metadata !{ i32 2, metadata !"foo", i32 50 }
; CHECK-NOT: module flag identifiers must be unique
!9 = metadata !{ i32 2, metadata !"bar", i32 51 }
!10 = metadata !{ i32 3, metadata !"bar", i32 51 }

!llvm.module.flags = !{
  !0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10 }
