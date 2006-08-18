; RUN: opt -analyze %s -tddatastructure

%str = type { int, int* }

implementation

void %bar(%str* %S, %str* %T) {
	%A1 = getelementptr %str* %S, long 0, ubyte 0
	%B1 = getelementptr %str* %S, long 0, ubyte 1
	%A2 = getelementptr %str* %S, long 0, ubyte 0
	%B2 = getelementptr %str* %S, long 0, ubyte 1
	%a1 = cast int* %A1 to long*
	%a2 = cast int* %A2 to long*
	%V = load long* %a1
	;store long %V, long* %a2
	%V2 = load int** %B1
	store int* %V2, int** %B2
	ret void
}
