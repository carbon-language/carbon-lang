; This checks to ensure that the inline pass deletes functions if they get 
; inlined into all of their callers.

; RUN: as < %s | opt -inline | dis | not grep %reallysmall

implementation

internal int %reallysmall(int %A) {
	ret int %A
}

void %caller1() {
	call int %reallysmall(int 5)
	ret void
}

void %caller2(int %A) {
	call int %reallysmall(int %A)
	ret void
}

int %caller3(int %A) {
	%B = call int %reallysmall(int %A)
	ret int %B
}
