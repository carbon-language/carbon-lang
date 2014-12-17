@ RUN: llvm-mc -triple armv7-linux-eabi -filetype asm -o /dev/null %s 2>&1 \
@ RUN:   | FileCheck %s

	.syntax unified
	.arm

	.global stm
	.type stm,%function
stm:
	stm sp!, {r0, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stm sp!, {r0, pc}
@ CHECK: ^
	stm r0!, {r0, sp}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stm r0!, {r0, sp}
@ CHECK: ^
	stm r1!, {r0, sp, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stm r1!, {r0, sp, pc}
@ CHECK: ^
	stm r2!, {sp, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stm r2!, {sp, pc}
@ CHECK: ^
	stm sp!, {pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stm sp!, {pc}
@ CHECK: ^
	stm r0!, {sp}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stm r0!, {sp}
@ CHECK: ^

	.global stmda
	.type stmda,%function
stmda:
	stmda sp!, {r0, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmda sp!, {r0, pc}
@ CHECK: ^
	stmda r0!, {r0, sp}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmda r0!, {r0, sp}
@ CHECK: ^
	stmda r1!, {r0, sp, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmda r1!, {r0, sp, pc}
@ CHECK: ^
	stmda r2!, {sp, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmda r2!, {sp, pc}
@ CHECK: ^
	stmda sp!, {pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmda sp!, {pc}
@ CHECK: ^
	stmda r0!, {sp}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmda r0!, {sp}
@ CHECK: ^

	.global stmdb
	.type stmdb,%function
stmdb:
	stmdb sp!, {r0, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmdb sp!, {r0, pc}
@ CHECK: ^
	stmdb r0!, {r0, sp}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmdb r0!, {r0, sp}
@ CHECK: ^
	stmdb r1!, {r0, sp, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmdb r1!, {r0, sp, pc}
@ CHECK: ^
	stmdb r2!, {sp, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmdb r2!, {sp, pc}
@ CHECK: ^
	stmdb sp!, {pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmdb sp!, {pc}
@ CHECK: ^
	stmdb r0!, {sp}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmdb r0!, {sp}
@ CHECK: ^

	.global stmib
	.type stmib,%function
stmib:
	stmib sp!, {r0, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmib sp!, {r0, pc}
@ CHECK: ^
	stmib r0!, {r0, sp}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmib r0!, {r0, sp}
@ CHECK: ^
	stmib r1!, {r0, sp, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmib r1!, {r0, sp, pc}
@ CHECK: ^
	stmib r2!, {sp, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmib r2!, {sp, pc}
@ CHECK: ^
	stmib sp!, {pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmib sp!, {pc}
@ CHECK: ^
	stmib r0!, {sp}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmib r0!, {sp}
@ CHECK: ^


	.global push
	.type push,%function
push:
	push {r0, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: push {r0, pc}
@ CHECK: ^
	push {r0, sp}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: push {r0, sp}
@ CHECK: ^
	push {r0, sp, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: push {r0, sp, pc}
@ CHECK: ^
	push {sp, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: push {sp, pc}
@ CHECK: ^
	push {pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: push {pc}
@ CHECK: ^
	push {sp}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: push {sp}
@ CHECK: ^

	.global single
	.type single,%function
single:
	stmdaeq r0, {r0}
@ CHECK-NOT: warning

