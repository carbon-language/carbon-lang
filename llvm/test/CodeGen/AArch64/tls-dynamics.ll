; RUN: llc -mtriple=aarch64-none-linux-gnu -relocation-model=pic -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -relocation-model=pic -filetype=obj < %s | llvm-objdump -r - | FileCheck --check-prefix=CHECK-RELOC %s

@general_dynamic_var = external thread_local global i32

define i32 @test_generaldynamic() {
; CHECK: test_generaldynamic:

  %val = load i32* @general_dynamic_var
  ret i32 %val

; CHECK: adrp x[[TLSDESC_HI:[0-9]+]], :tlsdesc:general_dynamic_var
; CHECK: add x0, x[[TLSDESC_HI]], #:tlsdesc_lo12:general_dynamic_var
; CHECK: ldr [[CALLEE:x[0-9]+]], [x[[TLSDESC_HI]], #:tlsdesc_lo12:general_dynamic_var]
; CHECK: .tlsdesccall general_dynamic_var
; CHECK-NEXT: blr [[CALLEE]]

; CHECK: mrs x[[TP:[0-9]+]], tpidr_el0
; CHECK: ldr w0, [x[[TP]], x0]

; CHECK-RELOC: R_AARCH64_TLSDESC_ADR_PAGE
; CHECK-RELOC: R_AARCH64_TLSDESC_ADD_LO12_NC
; CHECK-RELOC: R_AARCH64_TLSDESC_LD64_LO12_NC
; CHECK-RELOC: R_AARCH64_TLSDESC_CALL

}

define i32* @test_generaldynamic_addr() {
; CHECK: test_generaldynamic_addr:

  ret i32* @general_dynamic_var

; CHECK: adrp x[[TLSDESC_HI:[0-9]+]], :tlsdesc:general_dynamic_var
; CHECK: add x0, x[[TLSDESC_HI]], #:tlsdesc_lo12:general_dynamic_var
; CHECK: ldr [[CALLEE:x[0-9]+]], [x[[TLSDESC_HI]], #:tlsdesc_lo12:general_dynamic_var]
; CHECK: .tlsdesccall general_dynamic_var
; CHECK-NEXT: blr [[CALLEE]]

; CHECK: mrs [[TP:x[0-9]+]], tpidr_el0
; CHECK: add x0, [[TP]], x0

; CHECK-RELOC: R_AARCH64_TLSDESC_ADR_PAGE
; CHECK-RELOC: R_AARCH64_TLSDESC_ADD_LO12_NC
; CHECK-RELOC: R_AARCH64_TLSDESC_LD64_LO12_NC
; CHECK-RELOC: R_AARCH64_TLSDESC_CALL

}

@local_dynamic_var = external thread_local(localdynamic) global i32

define i32 @test_localdynamic() {
; CHECK: test_localdynamic:

  %val = load i32* @local_dynamic_var
  ret i32 %val

; CHECK: adrp x[[TLSDESC_HI:[0-9]+]], :tlsdesc:_TLS_MODULE_BASE_
; CHECK: add x0, x[[TLSDESC_HI]], #:tlsdesc_lo12:_TLS_MODULE_BASE_
; CHECK: ldr [[CALLEE:x[0-9]+]], [x[[TLSDESC_HI]], #:tlsdesc_lo12:_TLS_MODULE_BASE_]
; CHECK: .tlsdesccall _TLS_MODULE_BASE_
; CHECK-NEXT: blr [[CALLEE]]

; CHECK: movz [[DTP_OFFSET:x[0-9]+]], #:dtprel_g1:local_dynamic_var
; CHECK: movk [[DTP_OFFSET]], #:dtprel_g0_nc:local_dynamic_var

; CHECK: ldr w0, [x0, [[DTP_OFFSET]]]

; CHECK-RELOC: R_AARCH64_TLSDESC_ADR_PAGE
; CHECK-RELOC: R_AARCH64_TLSDESC_ADD_LO12_NC
; CHECK-RELOC: R_AARCH64_TLSDESC_LD64_LO12_NC
; CHECK-RELOC: R_AARCH64_TLSDESC_CALL

}

define i32* @test_localdynamic_addr() {
; CHECK: test_localdynamic_addr:

  ret i32* @local_dynamic_var

; CHECK: adrp x[[TLSDESC_HI:[0-9]+]], :tlsdesc:_TLS_MODULE_BASE_
; CHECK: add x0, x[[TLSDESC_HI]], #:tlsdesc_lo12:_TLS_MODULE_BASE_
; CHECK: ldr [[CALLEE:x[0-9]+]], [x[[TLSDESC_HI]], #:tlsdesc_lo12:_TLS_MODULE_BASE_]
; CHECK: .tlsdesccall _TLS_MODULE_BASE_
; CHECK-NEXT: blr [[CALLEE]]

; CHECK: movz [[DTP_OFFSET:x[0-9]+]], #:dtprel_g1:local_dynamic_var
; CHECK: movk [[DTP_OFFSET]], #:dtprel_g0_nc:local_dynamic_var

; CHECK: add x0, x0, [[DTP_OFFSET]]

; CHECK-RELOC: R_AARCH64_TLSDESC_ADR_PAGE
; CHECK-RELOC: R_AARCH64_TLSDESC_ADD_LO12_NC
; CHECK-RELOC: R_AARCH64_TLSDESC_LD64_LO12_NC
; CHECK-RELOC: R_AARCH64_TLSDESC_CALL

}

; The entire point of the local-dynamic access model is to have a single call to
; the expensive resolver. Make sure we achieve that goal.

@local_dynamic_var2 = external thread_local(localdynamic) global i32

define i32 @test_localdynamic_deduplicate() {
; CHECK: test_localdynamic_deduplicate:

  %val = load i32* @local_dynamic_var
  %val2 = load i32* @local_dynamic_var2

  %sum = add i32 %val, %val2
  ret i32 %sum

; CHECK: adrp x[[TLSDESC_HI:[0-9]+]], :tlsdesc:_TLS_MODULE_BASE_
; CHECK: add x0, x[[TLSDESC_HI]], #:tlsdesc_lo12:_TLS_MODULE_BASE_
; CHECK: ldr [[CALLEE:x[0-9]+]], [x[[TLSDESC_HI]], #:tlsdesc_lo12:_TLS_MODULE_BASE_]
; CHECK: .tlsdesccall _TLS_MODULE_BASE_
; CHECK-NEXT: blr [[CALLEE]]

; CHECK-NOT: _TLS_MODULE_BASE_

; CHECK: ret
}
