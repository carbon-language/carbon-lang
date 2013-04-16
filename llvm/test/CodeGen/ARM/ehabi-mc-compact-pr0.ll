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
; RUN:   | FileCheck %s --check-prefix=CHECK-RELOC

define void @_Z4testv() {
entry:
  tail call void @_Z15throw_exceptionv()
  ret void
}

declare void @_Z15throw_exceptionv()

; CHECK-NOT: section .ARM.extab
; CHECK: section .text
; CHECK-NOT: section .ARM.extab
; CHECK: section .ARM.exidx
; CHECK-NEXT: 0000 00000000 80849b80
; CHECK-NOT: section .ARM.extab

; CHECK-FP-ELIM-NOT: section .ARM.extab
; CHECK-FP-ELIM: section .text
; CHECK-FP-ELIM-NOT: section .ARM.extab
; CHECK-FP-ELIM: section .ARM.exidx
; CHECK-FP-ELIM-NEXT: 0000 00000000 b0808480
; CHECK-FP-ELIM-NOT: section .ARM.extab

; CHECK-RELOC: RELOCATION RECORDS FOR [.ARM.exidx]
; CHECK-RELOC-NEXT: 0 R_ARM_PREL31 .text
; CHECK-RELOC-NEXT: 0 R_ARM_NONE __aeabi_unwind_cpp_pr0
