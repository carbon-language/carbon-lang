; Make sure that functions are removed successfully if they are referred to by
; a global that is dead.  Make sure any globals they refer to die as well.

; RUN: if as < %s | opt -globaldce | dis | grep foo
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

%b = internal global int ()* %foo   ;; Unused, kills %foo

%foo = internal global int 7         ;; Should die when function %foo is killed

implementation

internal int %foo() {               ;; dies when %b dies.
	%ret = load int* %foo
	ret int %ret
}

