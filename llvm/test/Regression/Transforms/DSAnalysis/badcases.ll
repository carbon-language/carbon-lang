; This file contains a list of situations where node folding should happen...
;
; RUN: opt -analyze %s -tddatastructure

implementation

void %test1({int, int} * %X) {
	getelementptr {int, int} * %X, long 0
	%Y = cast {int, int} * %X to sbyte*
	%Z = getelementptr sbyte* %Y, long 7
	store sbyte 6, sbyte *%Z
	ret void
}

void %test2({int, int} * %X) {
	getelementptr {int, int} * %X, long 0
	%Y = cast {int, int} * %X to {sbyte,sbyte,sbyte,sbyte,sbyte,sbyte,sbyte,sbyte} *
	%Z = getelementptr {sbyte,sbyte,sbyte,sbyte,sbyte,sbyte,sbyte,sbyte}* %Y, long 0, ubyte 7
	store sbyte 6, sbyte *%Z
	ret void
}

