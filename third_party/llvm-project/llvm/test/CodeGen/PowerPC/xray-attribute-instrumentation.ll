; RUN: llc -filetype=asm -o - -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -filetype=asm -o - -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:    -relocation-model=pic < %s | FileCheck %s

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK-LABEL: foo:
; CHECK-NEXT:  .Lfunc_begin0:
; CHECK-LABEL: .Ltmp0:
; CHECK:              b .Ltmp1
; CHECK-NEXT:         nop
; CHECK-NEXT:         std 0, -8(1)
; CHECK-NEXT:         mflr 0
; CHECK-NEXT:         bl __xray_FunctionEntry
; CHECK-NEXT:         nop
; CHECK-NEXT:         mtlr 0
; CHECK-LABEL: .Ltmp1:
  ret i32 0
; CHECK-LABEL: .Ltmp2:
; CHECK:              blr
; CHECK-NEXT:         nop
; CHECK-NEXT:         std 0, -8(1)
; CHECK-NEXT:         mflr 0
; CHECK-NEXT:         bl __xray_FunctionExit
; CHECK-NEXT:         nop
; CHECK-NEXT:         mtlr 0
}
; CHECK-LABEL: xray_instr_map,"ao",@progbits,foo{{$}}
; CHECK:      .Lxray_sleds_start0:
; CHECK-NEXT: .Ltmp3:
; CHECK-NEXT:         .quad   .Ltmp0-.Ltmp3
; CHECK-NEXT:         .quad   .Lfunc_begin0-(.Ltmp3+8)
; CHECK-NEXT:         .byte   0x00
; CHECK-NEXT:         .byte   0x01
; CHECK-NEXT:         .byte   0x02
; CHECK-NEXT:         .space  13
; CHECK-NEXT: .Ltmp4:
; CHECK-NEXT:         .quad   .Ltmp2-.Ltmp4
; CHECK-NEXT:         .quad   .Lfunc_begin0-(.Ltmp4+8)
; CHECK-NEXT:         .byte   0x01
; CHECK-NEXT:         .byte   0x01
; CHECK-NEXT:         .byte   0x02
; CHECK-NEXT:         .space  13
; CHECK-NEXT: .Lxray_sleds_end0:
; CHECK-LABEL: xray_fn_idx,"awo",@progbits,foo{{$}}
; CHECK:              .p2align        4
; CHECK-NEXT:         .quad   .Lxray_sleds_start0
; CHECK-NEXT:         .quad   .Lxray_sleds_end0
; CHECK-NEXT:         .text
