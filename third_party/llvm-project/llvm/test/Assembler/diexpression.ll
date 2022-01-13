; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{
; CHECK-SAME: !DIExpression(),
; CHECK-SAME: !DIExpression(DW_OP_deref),
; CHECK-SAME: !DIExpression(DW_OP_constu, 3, DW_OP_plus),
; CHECK-SAME: !DIExpression(DW_OP_LLVM_fragment, 3, 7),
; CHECK-SAME: !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 3, DW_OP_LLVM_fragment, 3, 7),
; CHECK-SAME: !DIExpression(DW_OP_constu, 2, DW_OP_swap, DW_OP_xderef),
; CHECK-SAME: !DIExpression(DW_OP_plus_uconst, 3)
; CHECK-SAME: !DIExpression(DW_OP_LLVM_convert, 16, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_signed)
; CHECK-SAME: !DIExpression(DW_OP_LLVM_tag_offset, 1)}

!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}

!0 = !DIExpression()
!1 = !DIExpression(DW_OP_deref)
!2 = !DIExpression(DW_OP_constu, 3, DW_OP_plus)
!3 = !DIExpression(DW_OP_LLVM_fragment, 3, 7)
!4 = !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 3, DW_OP_LLVM_fragment, 3, 7)
!5 = !DIExpression(DW_OP_constu, 2, DW_OP_swap, DW_OP_xderef)
!6 = !DIExpression(DW_OP_plus_uconst, 3)
!7 = !DIExpression(DW_OP_LLVM_convert, 16, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_signed)
!8 = !DIExpression(DW_OP_LLVM_tag_offset, 1)
