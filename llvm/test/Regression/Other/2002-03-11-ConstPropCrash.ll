; When constant propogating terminator instructions, the basic block iterator
; was not updated to refer to the final position of the new terminator.  This
; can be bad, f.e. because constproping a terminator can lead to the 
; destruction of PHI nodes, which invalidates the iterator!
;
; Fixed by adding new arguments to ConstantFoldTerminator
;
; RUN: llvm-as < %s | opt -constprop

implementation

void "build_tree"(int %ml)
begin
	br label %bb2

bb2:
	%reg137 = phi int [ %reg140, %bb2 ], [ 12, %0 ]		; <int> [#uses=2]
	%reg138 = phi uint [ %reg139, %bb2 ], [ 0, %0 ]		; <uint> [#uses=3]
	%reg139 = add uint %reg138, 1		; <uint> [#uses=1]
	%reg140 = add int %reg137, -1		; <int> [#uses=1]
	br bool false, label %bb2, label %bb3

bb3:
	ret void
end
