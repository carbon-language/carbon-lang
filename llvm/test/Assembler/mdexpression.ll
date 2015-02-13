; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1, !2, !3, !4}
!named = !{!0, !1, !2, !3, !4}

; CHECK:      !0 = !MDExpression()
; CHECK-NEXT: !1 = !MDExpression(DW_OP_deref)
; CHECK-NEXT: !2 = !MDExpression(DW_OP_plus, 3)
; CHECK-NEXT: !3 = !MDExpression(DW_OP_bit_piece, 3, 7)
; CHECK-NEXT: !4 = !MDExpression(DW_OP_deref, DW_OP_plus, 3, DW_OP_bit_piece, 3, 7)
!0 = !MDExpression()
!1 = !MDExpression(DW_OP_deref)
!2 = !MDExpression(DW_OP_plus, 3)
!3 = !MDExpression(DW_OP_bit_piece, 3, 7)
!4 = !MDExpression(DW_OP_deref, DW_OP_plus, 3, DW_OP_bit_piece, 3, 7)
