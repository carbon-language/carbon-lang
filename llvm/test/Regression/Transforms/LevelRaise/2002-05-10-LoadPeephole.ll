; This testcase should have the cast propogated through the load
; just like a store does...
;
; RUN: llvm-as < %s | opt -raise | llvm-dis | grep ' cast ' | not grep '*'

int "test"(uint * %Ptr) {
	%P2 = cast uint *%Ptr to int *
	%Val = load int * %P2
	ret int %Val
}
