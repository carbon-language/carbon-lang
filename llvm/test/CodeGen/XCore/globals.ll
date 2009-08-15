; RUN: llvm-as < %s | llc -march=xcore -mcpu=xs1b-generic | FileCheck %s

define i32 *@addr_G1() {
entry:
; CHECK: addr_G1:
; CHECK: ldaw r0, dp[G1]
	ret i32* @G1
}

define i32 *@addr_G2() {
entry:
; CHECK: addr_G2:
; CHECK: ldaw r0, dp[G2]
	ret i32* @G2
}

define i32 *@addr_G3() {
entry:
; CHECK: addr_G3:
; CHECK: ldaw r11, cp[G3]
; CHECK: mov r0, r11
	ret i32* @G3
}

@G1 = global i32 4712
; CHECK: .section .dp.data,"awd",@progbits
; CHECK: G1:

@G2 = global i32 0
; CHECK: .section .dp.bss,"awd",@nobits
; CHECK: G2:

@G3 = constant i32 9401
; CHECK: .section .cp.rodata,"ac",@progbits
; CHECK: G3:

