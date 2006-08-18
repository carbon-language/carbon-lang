; This file contains a list of cases where node folding should NOT happen
;
; RUN: opt -analyze %s -tddatastructure
;

implementation

void %test1({int, int}* %X) {
        getelementptr {int, int} * %X, long 0
        %Y = cast {int, int}* %X to uint*
        store uint 5, uint* %Y
        ret void
}

; Test that "structural" equality works.  Pointers can land in pointers n 
; stuff.
void %test2({int*, int*}* %X) {
	getelementptr {int*, int*}* %X, long 0
	%Y = cast {int*, int*}* %X to {uint*, long*}*
	getelementptr {uint*, long*}* %Y, long 0
	ret void
}

