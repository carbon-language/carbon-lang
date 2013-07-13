; RUN: llc -mtriple=aarch64-none-linux-gnu -relocation-model=pic < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -relocation-model=pic -filetype=obj < %s | llvm-objdump -r - | FileCheck --check-prefix=CHECK-ELF %s

@var_simple = hidden global i32 0
@var_got = global i32 0
@var_tlsgd = thread_local global i32 0
@var_tlsld = thread_local(localdynamic) global i32 0
@var_tlsie = thread_local(initialexec) global i32 0
@var_tlsle = thread_local(localexec) global i32 0

define void @test_inline_modifier_L() nounwind {
; CHECK-LABEL: test_inline_modifier_L:
  call void asm sideeffect "add x0, x0, ${0:L}", "S,~{x0}"(i32* @var_simple)
  call void asm sideeffect "ldr x0, [x0, ${0:L}]", "S,~{x0}"(i32* @var_got)
  call void asm sideeffect "add x0, x0, ${0:L}", "S,~{x0}"(i32* @var_tlsgd)
  call void asm sideeffect "add x0, x0, ${0:L}", "S,~{x0}"(i32* @var_tlsld)
  call void asm sideeffect "ldr x0, [x0, ${0:L}]", "S,~{x0}"(i32* @var_tlsie)
  call void asm sideeffect "add x0, x0, ${0:L}", "S,~{x0}"(i32* @var_tlsle)
; CHECK: add x0, x0, #:lo12:var_simple
; CHECK: ldr x0, [x0, #:got_lo12:var_got]
; CHECK: add x0, x0, #:tlsdesc_lo12:var_tlsgd
; CHECK: add x0, x0, #:dtprel_lo12:var_tlsld
; CHECK: ldr x0, [x0, #:gottprel_lo12:var_tlsie]
; CHECK: add x0, x0, #:tprel_lo12:var_tlsle

; CHECK-ELF: R_AARCH64_ADD_ABS_LO12_NC var_simple
; CHECK-ELF: R_AARCH64_LD64_GOT_LO12_NC var_got
; CHECK-ELF: R_AARCH64_TLSDESC_ADD_LO12_NC var_tlsgd
; CHECK-ELF: R_AARCH64_TLSLD_ADD_DTPREL_LO12 var_tlsld
; CHECK-ELF: R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC var_tlsie
; CHECK-ELF: R_AARCH64_TLSLE_ADD_TPREL_LO12 var_tlsle

  ret void
}

define void @test_inline_modifier_G() nounwind {
; CHECK-LABEL: test_inline_modifier_G:
  call void asm sideeffect "add x0, x0, ${0:G}, lsl #12", "S,~{x0}"(i32* @var_tlsld)
  call void asm sideeffect "add x0, x0, ${0:G}, lsl #12", "S,~{x0}"(i32* @var_tlsle)
; CHECK: add x0, x0, #:dtprel_hi12:var_tlsld, lsl #12
; CHECK: add x0, x0, #:tprel_hi12:var_tlsle, lsl #12

; CHECK-ELF: R_AARCH64_TLSLD_ADD_DTPREL_HI12 var_tlsld
; CHECK-ELF: R_AARCH64_TLSLE_ADD_TPREL_HI12 var_tlsle

  ret void
}

define void @test_inline_modifier_A() nounwind {
; CHECK-LABEL: test_inline_modifier_A:
  call void asm sideeffect "adrp x0, ${0:A}", "S,~{x0}"(i32* @var_simple)
  call void asm sideeffect "adrp x0, ${0:A}", "S,~{x0}"(i32* @var_got)
  call void asm sideeffect "adrp x0, ${0:A}", "S,~{x0}"(i32* @var_tlsgd)
  call void asm sideeffect "adrp x0, ${0:A}", "S,~{x0}"(i32* @var_tlsie)
  ; N.b. All tprel and dtprel relocs are modified: lo12 or granules.
; CHECK: adrp x0, var_simple
; CHECK: adrp x0, :got:var_got
; CHECK: adrp x0, :tlsdesc:var_tlsgd
; CHECK: adrp x0, :gottprel:var_tlsie

; CHECK-ELF: R_AARCH64_ADR_PREL_PG_HI21 var_simple
; CHECK-ELF: R_AARCH64_ADR_GOT_PAGE var_got
; CHECK-ELF: R_AARCH64_TLSDESC_ADR_PAGE var_tlsgd
; CHECK-ELF: R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 var_tlsie

  ret void
}

define void @test_inline_modifier_wx(i32 %small, i64 %big) nounwind {
; CHECK-LABEL: test_inline_modifier_wx:
  call i32 asm sideeffect "add $0, $0, $0", "=r,0"(i32 %small)
  call i32 asm sideeffect "add ${0:w}, ${0:w}, ${0:w}", "=r,0"(i32 %small)
  call i32 asm sideeffect "add ${0:x}, ${0:x}, ${0:x}", "=r,0"(i32 %small)
; CHECK: //APP
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}

  call i64 asm sideeffect "add $0, $0, $0", "=r,0"(i64 %big)
  call i64 asm sideeffect "add ${0:w}, ${0:w}, ${0:w}", "=r,0"(i64 %big)
  call i64 asm sideeffect "add ${0:x}, ${0:x}, ${0:x}", "=r,0"(i64 %big)
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}

  call i32 asm sideeffect "add ${0:w}, ${1:w}, ${1:w}", "=r,r"(i32 0)
  call i32 asm sideeffect "add ${0:x}, ${1:x}, ${1:x}", "=r,r"(i32 0)
; CHECK: add {{w[0-9]+}}, wzr, wzr
; CHECK: add {{x[0-9]+}}, xzr, xzr
  ret void
}

define void @test_inline_modifier_bhsdq() nounwind {
; CHECK-LABEL: test_inline_modifier_bhsdq:
  call float asm sideeffect "ldr ${0:b}, [sp]", "=w"()
  call float asm sideeffect "ldr ${0:h}, [sp]", "=w"()
  call float asm sideeffect "ldr ${0:s}, [sp]", "=w"()
  call float asm sideeffect "ldr ${0:d}, [sp]", "=w"()
  call float asm sideeffect "ldr ${0:q}, [sp]", "=w"()
; CHECK: ldr b0, [sp]
; CHECK: ldr h0, [sp]
; CHECK: ldr s0, [sp]
; CHECK: ldr d0, [sp]
; CHECK: ldr q0, [sp]

  call double asm sideeffect "ldr ${0:b}, [sp]", "=w"()
  call double asm sideeffect "ldr ${0:h}, [sp]", "=w"()
  call double asm sideeffect "ldr ${0:s}, [sp]", "=w"()
  call double asm sideeffect "ldr ${0:d}, [sp]", "=w"()
  call double asm sideeffect "ldr ${0:q}, [sp]", "=w"()
; CHECK: ldr b0, [sp]
; CHECK: ldr h0, [sp]
; CHECK: ldr s0, [sp]
; CHECK: ldr d0, [sp]
; CHECK: ldr q0, [sp]
  ret void
}

define void @test_inline_modifier_c() nounwind {
; CHECK-LABEL: test_inline_modifier_c:
  call void asm sideeffect "adr x0, ${0:c}", "i"(i32 3)
; CHECK: adr x0, 3

  ret void
}
