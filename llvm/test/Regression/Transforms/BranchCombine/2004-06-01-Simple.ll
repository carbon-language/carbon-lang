; Basic test of -branch-combine functionality
; RUN: llvm-as < %s | opt -branch-combine | llvm-dis | egrep 'newCommon:.*; preds =.*no_exit.1' | grep loopentry.2

target endian = big
target pointersize = 64

implementation   ; Functions:

void %main() {
entry:
	br bool false, label %__main.entry, label %endif.0.i

endif.0.i:		; preds = %entry
	ret void

__main.entry:		; preds = %entry
	br label %no_exit.1

no_exit.1:		; preds = %__main.entry, %no_exit.1, %loopentry.2
	br bool false, label %loopentry.2, label %no_exit.1

loopentry.2:		; preds = %no_exit.1
	br label %no_exit.1
}
