; Due to a recent change, this testcase now sends the raise pass into an infinite loop
;
; RUN: as < %s | opt -raise

implementation

void %test(sbyte* %P, void(...) * %F) {
	call void (...)* %F(sbyte* %P)
	ret void
}
