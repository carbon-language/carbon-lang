; RUN: llc < %s -march=bfin

; This test tries to use a JustCC register as a data operand for MOVEcc.  It
; calls copyRegToReg(JustCC -> DP), failing because JustCC can only be copied to
; D.  The proper solution would be to restrict the virtual register to D only.

define i32 @main() {
entry:
	br label %loopentry

loopentry:
	%done = icmp sle i64 undef, 5
	br i1 %done, label %loopentry, label %exit.1

exit.1:
	ret i32 0
}
