; The inliner is breaking inlining invoke instructions where there is a PHI 
; node in the exception destination, and the inlined function contains an 
; unwind instruction.

; RUN: llvm-as < %s | opt -inline -disable-output

implementation

linkonce void %foo() {
  unwind
}

int %test() {
BB1:
	invoke void %foo() to label %Cont except label %Cont
Cont:
	%A = phi int [ 0, %BB1], [0, %BB1]
	ret int %A
}
