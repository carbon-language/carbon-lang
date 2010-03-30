; RUN: llc < %s -march=pic16 | FileCheck %s
; XFAIL: vg_leak

@G1 = global i32 4712, section "Address=412"
; CHECK: @G1.412..user_section.#	IDATA	412
; CHECK: @G1
; CHECK:     dl 4712

@G2 = global i32 0, section "Address=412"
; CHECK: @G2.412..user_section.#	UDATA	412
; CHECK: @G2 RES 4

@G3 = addrspace(1) constant i32 4712, section "Address=412"
; CHECK: @G3.412..user_section.#	ROMDATA	412
; CHECK: @G3
; CHECK:     rom_dl 4712


