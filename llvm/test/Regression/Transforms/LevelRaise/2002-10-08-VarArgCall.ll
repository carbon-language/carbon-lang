; RUN: if as < %s | opt -raise | dis | grep call | grep \.\.\.
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

implementation

void %test(sbyte* %P) {
	%Q = cast sbyte* %P to void (...)*
	call void (...)* %Q(sbyte* %P)
	ret void
}
