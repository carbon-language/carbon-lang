; This should cause the global node to collapse!!
;
; RUN: analyze %s -datastructure-gc --dsgc-check-flags=test:GAU

%Tree = type { int, %Tree*, %Tree* }
%T5 = external global %Tree

implementation   ; Functions:

void %makeMore(%Tree** %X) {
	store %Tree* cast (long add (long cast (%Tree* %T5 to long), long 5) to %Tree*), %Tree** %X
	%test = load %Tree** %X
	ret void
}

