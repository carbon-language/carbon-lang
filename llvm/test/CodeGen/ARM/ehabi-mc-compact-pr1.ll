; RUN: llc -mtriple armv7-unknown-linux-gnueabi \
; RUN:     -arm-enable-ehabi -arm-enable-ehabi-descriptors \
; RUN:     -disable-fp-elim -filetype=obj -o - %s \
; RUN:   | llvm-objdump -s - \
; RUN:   | FileCheck %s --check-prefix=CHECK

; RUN: llc -mtriple armv7-unknown-linux-gnueabi \
; RUN:     -arm-enable-ehabi -arm-enable-ehabi-descriptors \
; RUN:     -filetype=obj -o - %s \
; RUN:   | llvm-objdump -s - \
; RUN:   | FileCheck %s --check-prefix=CHECK-FP-ELIM

; RUN: llc -mtriple armv7-unknown-linux-gnueabi \
; RUN:     -arm-enable-ehabi -arm-enable-ehabi-descriptors \
; RUN:     -disable-fp-elim -filetype=obj -o - %s \
; RUN:   | llvm-objdump -r - \
; RUN:   | FileCheck %s --check-prefix=CHECK-RELOC

; RUN: llc -mtriple armv7-unknown-linux-gnueabi \
; RUN:     -arm-enable-ehabi -arm-enable-ehabi-descriptors \
; RUN:     -filetype=obj -o - %s \
; RUN:   | llvm-objdump -r - \
; RUN:   | FileCheck %s --check-prefix=CHECK-FP-ELIM-RELOC

define i32 @_Z3addiiiiiiii(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h) {
entry:
  %add = add nsw i32 %b, %a
  %add1 = add nsw i32 %add, %c
  %add2 = add nsw i32 %add1, %d
  tail call void @_Z15throw_exceptioni(i32 %add2)
  %add3 = add nsw i32 %f, %e
  %add4 = add nsw i32 %add3, %g
  %add5 = add nsw i32 %add4, %h
  tail call void @_Z15throw_exceptioni(i32 %add5)
  %add6 = add nsw i32 %add5, %add2
  ret i32 %add6
}

declare void @_Z15throw_exceptioni(i32)

; CHECK-NOT: section .ARM.extab
; CHECK: section .text
; CHECK: section .ARM.extab
; CHECK-NEXT: 0000 419b0181 b0b08384
; CHECK: section .ARM.exidx
; CHECK-NEXT: 0000 00000000 00000000
; CHECK-NOT: section .ARM.extab

; CHECK-FP-ELIM-NOT: section .ARM.extab
; CHECK-FP-ELIM: section .text
; CHECK-FP-ELIM-NOT: section .ARM.extab
; CHECK-FP-ELIM: section .ARM.exidx
; CHECK-FP-ELIM-NEXT: 0000 00000000 b0838480
; CHECK-FP-ELIM-NOT: section .ARM.extab

; CHECK-RELOC: RELOCATION RECORDS FOR [.ARM.exidx]
; CHECK-RELOC-NEXT: 0 R_ARM_PREL31 .text
; CHECK-RELOC-NEXT: 0 R_ARM_NONE __aeabi_unwind_cpp_pr1

; CHECK-FP-ELIM-RELOC: RELOCATION RECORDS FOR [.ARM.exidx]
; CHECK-FP-ELIM-RELOC-NEXT: 0 R_ARM_PREL31 .text
; CHECK-FP-ELIM-RELOC-NEXT: 0 R_ARM_NONE __aeabi_unwind_cpp_pr0
