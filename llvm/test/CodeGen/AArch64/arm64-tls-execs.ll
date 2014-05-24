; RUN: llc -mtriple=arm64-none-linux-gnu -verify-machineinstrs -show-mc-encoding < %s | FileCheck %s
; RUN: llc -mtriple=arm64-none-linux-gnu -filetype=obj < %s | llvm-objdump -r - | FileCheck --check-prefix=CHECK-RELOC %s

@initial_exec_var = external thread_local(initialexec) global i32

define i32 @test_initial_exec() {
; CHECK-LABEL: test_initial_exec:
  %val = load i32* @initial_exec_var

; CHECK: adrp x[[GOTADDR:[0-9]+]], :gottprel:initial_exec_var
; CHECK: ldr x[[TP_OFFSET:[0-9]+]], [x[[GOTADDR]], :gottprel_lo12:initial_exec_var]
; CHECK: mrs x[[TP:[0-9]+]], TPIDR_EL0
; CHECK: ldr w0, [x[[TP]], x[[TP_OFFSET]]]

; CHECK-RELOC: R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21
; CHECK-RELOC: R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC

  ret i32 %val
}

define i32* @test_initial_exec_addr() {
; CHECK-LABEL: test_initial_exec_addr:
  ret i32* @initial_exec_var

; CHECK: adrp x[[GOTADDR:[0-9]+]], :gottprel:initial_exec_var
; CHECK: ldr [[TP_OFFSET:x[0-9]+]], [x[[GOTADDR]], :gottprel_lo12:initial_exec_var]
; CHECK: mrs [[TP:x[0-9]+]], TPIDR_EL0
; CHECK: add x0, [[TP]], [[TP_OFFSET]]

; CHECK-RELOC: R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21
; CHECK-RELOC: R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC

}

@local_exec_var = thread_local(localexec) global i32 0

define i32 @test_local_exec() {
; CHECK-LABEL: test_local_exec:
  %val = load i32* @local_exec_var

; CHECK: movz [[TP_OFFSET:x[0-9]+]], #:tprel_g1:local_exec_var // encoding: [0bAAA{{[01]+}},A,0b101AAAAA,0x92]
; CHECK: movk [[TP_OFFSET]], #:tprel_g0_nc:local_exec_var
; CHECK: mrs x[[TP:[0-9]+]], TPIDR_EL0
; CHECK: ldr w0, [x[[TP]], [[TP_OFFSET]]]

; CHECK-RELOC: R_AARCH64_TLSLE_MOVW_TPREL_G1
; CHECK-RELOC: R_AARCH64_TLSLE_MOVW_TPREL_G0_NC

  ret i32 %val
}

define i32* @test_local_exec_addr() {
; CHECK-LABEL: test_local_exec_addr:
  ret i32* @local_exec_var

; CHECK: movz [[TP_OFFSET:x[0-9]+]], #:tprel_g1:local_exec_var
; CHECK: movk [[TP_OFFSET]], #:tprel_g0_nc:local_exec_var
; CHECK: mrs [[TP:x[0-9]+]], TPIDR_EL0
; CHECK: add x0, [[TP]], [[TP_OFFSET]]

; CHECK-RELOC: R_AARCH64_TLSLE_MOVW_TPREL_G1
; CHECK-RELOC: R_AARCH64_TLSLE_MOVW_TPREL_G0_NC
}
