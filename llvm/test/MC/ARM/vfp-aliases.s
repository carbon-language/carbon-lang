@ RUN: llvm-mc -triple armv7-eabi -filetype asm -o - %s | FileCheck %s

	.syntax unified
	.fpu vfp

	.type aliases,%function
aliases:
	fstmfdd sp!, {d0}
	fstmead sp!, {d0}
	fstmdbd sp!, {d0}
	fstmiad sp!, {d0}
	fstmfds sp!, {s0}
	fstmeas sp!, {s0}
	fstmdbs sp!, {s0}
	fstmias sp!, {s0}

	fldmias sp!, {s0}
	fldmdbs sp!, {s0}
	fldmeas sp!, {s0}
	fldmfds sp!, {s0}
	fldmiad sp!, {d0}
	fldmdbd sp!, {d0}
	fldmead sp!, {d0}
	fldmfdd sp!, {d0}

	fstmeax sp!, {d0}
	fldmfdx sp!, {d0}

	fstmfdx sp!, {d0}
	fldmeax sp!, {d0}

@ CHECK-LABEL: aliases
@ CHECK: 	vpush {d0}
@ CHECK: 	vstmia sp!, {d0}
@ CHECK: 	vpush {d0}
@ CHECK: 	vstmia sp!, {d0}
@ CHECK: 	vpush {s0}
@ CHECK: 	vstmia sp!, {s0}
@ CHECK: 	vpush {s0}
@ CHECK: 	vstmia sp!, {s0}
@ CHECK: 	vpop {s0}
@ CHECK: 	vldmdb sp!, {s0}
@ CHECK: 	vldmdb sp!, {s0}
@ CHECK: 	vpop {s0}
@ CHECK: 	vpop {d0}
@ CHECK: 	vldmdb sp!, {d0}
@ CHECK: 	vldmdb sp!, {d0}
@ CHECK: 	vpop {d0}
@ CHECK: 	fstmiax sp!, {d0}
@ CHECK: 	fldmiax sp!, {d0}
@ CHECK: 	fstmdbx sp!, {d0}
@ CHECK: 	fldmdbx sp!, {d0}

