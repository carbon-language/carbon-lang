; This test makes sure that these instructions are properly eliminated.
;

; RUN: if as < %s | opt -instcombine | dis | grep sh
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

implementation

int "test1"(int %A) {
	%B = shl int %A, ubyte 0
	ret int %B
}

int "test2"(ubyte %A) {
	%B = shl int 0, ubyte %A
	ret int %B
}

int "test3"(int %A) {
	%B = shr int %A, ubyte 0
	ret int %B
}

int "test4"(ubyte %A) {
	%B = shr int 0, ubyte %A
	ret int %B
}

uint "test5"(uint %A) {
	%B = shr uint %A, ubyte 32  ;; shift all bits out
	ret uint %B
}

