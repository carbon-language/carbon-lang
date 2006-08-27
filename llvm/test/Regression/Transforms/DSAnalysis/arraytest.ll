;
; RUN: llvm-as < %s | opt -analyze -tddatastructure
%crazy = type [2 x { [2 x sbyte], short } ]

implementation

sbyte *%test1(%crazy* %P1) {    ; No merging, constant indexing
	%P = getelementptr %crazy* %P1, long 0, long 0, ubyte 0, long 1
	ret sbyte *%P
}

sbyte *%test2(%crazy* %P1) {    ; No merging, constant indexing
	%P = getelementptr %crazy* %P1, long 0, long 1, ubyte 0, long 0
	ret sbyte *%P
}

sbyte *%test3(%crazy* %P1) {    ; No merging, constant indexing, must handle outter index
	%P = getelementptr %crazy* %P1, long -1, long 0, ubyte 0, long 0
	ret sbyte *%P
}

sbyte *%mtest1(%crazy* %P1, long %idx) {    ; Merging deepest array
	%P = getelementptr %crazy* %P1, long 0, long 0, ubyte 0, long %idx
	ret sbyte *%P
}
sbyte *%mtest2(%crazy* %P1, long %idx) {    ; Merge top array
	%P = getelementptr %crazy* %P1, long 0, long %idx, ubyte 0, long 1
	ret sbyte *%P
}
sbyte *%mtest3(%crazy* %P1, long %idx) {    ; Merge array %crazy is in
	%P = getelementptr %crazy* %P1, long %idx, long 0, ubyte 0, long 1
	ret sbyte *%P
}

sbyte *%m2test1(%crazy* %P1, long %idx) {    ; Merge two arrays
	%P = getelementptr %crazy* %P1, long 0, long %idx, ubyte 0, long %idx
	ret sbyte *%P
}

