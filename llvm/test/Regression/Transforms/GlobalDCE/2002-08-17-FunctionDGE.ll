; Make sure that functions are removed successfully if they are referred to by
; a global that is dead.  Make sure any globals they refer to die as well.

; RUN: as < %s | opt -globaldce | dis | not grep foo

%b = internal global int ()* %foo   ;; Unused, kills %foo

%foo = internal global int 7         ;; Should die when function %foo is killed

implementation

internal int %foo() {               ;; dies when %b dies.
	%ret = load int* %foo
	ret int %ret
}

