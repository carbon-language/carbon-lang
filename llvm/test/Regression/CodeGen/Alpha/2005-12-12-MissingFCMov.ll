; This shouldn't crash
; RUN: llvm-as < %s | llc -march=alpha

; ModuleID = 'bugpoint-reduced-simplified.bc'
target endian = little
target pointersize = 64
%.str_4 = external global [44 x sbyte]		; <[44 x sbyte]*> [#uses=0]

implementation   ; Functions:

declare void %printf(int, ...)

void %main() {
entry:
	%tmp.11861 = setlt long 0, 1		; <bool> [#uses=1]
	%tmp.19466 = setlt long 0, 1		; <bool> [#uses=1]
	%tmp.21571 = setlt long 0, 1		; <bool> [#uses=1]
	%tmp.36796 = setlt long 0, 1		; <bool> [#uses=1]
	br bool %tmp.11861, label %loopexit.2, label %no_exit.2

no_exit.2:		; preds = %entry
	ret void

loopexit.2:		; preds = %entry
	br bool %tmp.19466, label %loopexit.3, label %no_exit.3.preheader

no_exit.3.preheader:		; preds = %loopexit.2
	ret void

loopexit.3:		; preds = %loopexit.2
	br bool %tmp.21571, label %no_exit.6, label %no_exit.4

no_exit.4:		; preds = %loopexit.3
	ret void

no_exit.6:		; preds = %no_exit.6, %loopexit.3
	%tmp.30793 = setgt long 0, 0		; <bool> [#uses=1]
	br bool %tmp.30793, label %loopexit.6, label %no_exit.6

loopexit.6:		; preds = %no_exit.6
	%Z.1 = select bool %tmp.36796, double 1.000000e+00, double 0x3FEFFF7CEDE74EAE		; <double> [#uses=2]
	tail call void (int, ...)* %printf( int 0, long 0, long 0, long 0, double 1.000000e+00, double 1.000000e+00, double %Z.1, double %Z.1 )
	ret void
}
