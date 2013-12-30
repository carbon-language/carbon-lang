@ RUN: not llvm-mc -triple armv7-eabi -filetype asm -o /dev/null %s 2>&1 \
@ RUN:   | FileCheck %s

	.syntax unified
	.fpu vfp

	.type aliases,%function
aliases:
	fstmfdd sp!, {s0}
	fstmead sp!, {s0}
	fstmdbd sp!, {s0}
	fstmiad sp!, {s0}
	fstmfds sp!, {d0}
	fstmeas sp!, {d0}
	fstmdbs sp!, {d0}
	fstmias sp!, {d0}

	fldmias sp!, {d0}
	fldmdbs sp!, {d0}
	fldmeas sp!, {d0}
	fldmfds sp!, {d0}
	fldmiad sp!, {s0}
	fldmdbd sp!, {s0}
	fldmead sp!, {s0}
	fldmfdd sp!, {s0}

	fstmeax sp!, {s0}
	fldmfdx sp!, {s0}

	fstmfdx sp!, {s0}
	fldmeax sp!, {s0}

@ CHECK-LABEL: aliases
@ CHECK: error: VFP/Neon double precision register expected
@ CHECK:	fstmfdd sp!, {s0}
@ CHECK:                     ^
@ CHECK: error: VFP/Neon double precision register expected
@ CHECK:	fstmead sp!, {s0}
@ CHECK:                     ^
@ CHECK: error: VFP/Neon double precision register expected
@ CHECK:	fstmdbd sp!, {s0}
@ CHECK:                     ^
@ CHECK: error: VFP/Neon double precision register expected
@ CHECK:	fstmiad sp!, {s0}
@ CHECK:                     ^
@ CHECK: error: VFP/Neon single precision register expected
@ CHECK:	fstmfds sp!, {d0}
@ CHECK:                     ^
@ CHECK: error: VFP/Neon single precision register expected
@ CHECK:	fstmeas sp!, {d0}
@ CHECK:                     ^
@ CHECK: error: VFP/Neon single precision register expected
@ CHECK:	fstmdbs sp!, {d0}
@ CHECK:                     ^
@ CHECK: error: VFP/Neon single precision register expected
@ CHECK:	fstmias sp!, {d0}
@ CHECK:                     ^

@ CHECK: error: VFP/Neon single precision register expected
@ CHECK:	fldmias sp!, {d0}
@ CHECK:                     ^
@ CHECK: error: VFP/Neon single precision register expected
@ CHECK:	fldmdbs sp!, {d0}
@ CHECK:                     ^
@ CHECK: error: VFP/Neon single precision register expected
@ CHECK:	fldmeas sp!, {d0}
@ CHECK:                     ^
@ CHECK: error: VFP/Neon single precision register expected
@ CHECK:	fldmfds sp!, {d0}
@ CHECK:                     ^
@ CHECK: error: VFP/Neon double precision register expected
@ CHECK:	fldmiad sp!, {s0}
@ CHECK:                     ^
@ CHECK: error: VFP/Neon double precision register expected
@ CHECK:	fldmdbd sp!, {s0}
@ CHECK:                     ^
@ CHECK: error: VFP/Neon double precision register expected
@ CHECK:	fldmead sp!, {s0}
@ CHECK:                     ^
@ CHECK: error: VFP/Neon double precision register expected
@ CHECK:	fldmfdd sp!, {s0}
@ CHECK:                     ^

@ CHECK: error: VFP/Neon double precision register expected
@ CHECK:	fstmeax sp!, {s0}
@ CHECK:                     ^
@ CHECK: error: VFP/Neon double precision register expected
@ CHECK:	fldmfdx sp!, {s0}
@ CHECK:                     ^

@ CHECK: error: VFP/Neon double precision register expected
@ CHECK:	fstmfdx sp!, {s0}
@ CHECK:                     ^
@ CHECK: error: VFP/Neon double precision register expected
@ CHECK:	fldmeax sp!, {s0}
@ CHECK:                     ^

	fstmiaxcs r0, {s0}
	fstmiaxhs r0, {s0}
	fstmiaxls r0, {s0}
	fstmiaxvs r0, {s0}
@ CHECK: error: VFP/Neon double precision register expected
@ CHECK: 	fstmiaxcs r0, {s0}
@ CHECK:                      ^
@ CHECK: error: VFP/Neon double precision register expected
@ CHECK: 	fstmiaxhs r0, {s0}
@ CHECK:                      ^
@ CHECK: error: VFP/Neon double precision register expected
@ CHECK: 	fstmiaxls r0, {s0}
@ CHECK:                      ^
@ CHECK: error: VFP/Neon double precision register expected
@ CHECK: 	fstmiaxvs r0, {s0}
@ CHECK:                      ^

