; Expression analysis had a problem where the following assertion would get
; emitted:
; Constants.cpp:114: failed assertion `isValueValidForType(Ty, V) &&
;                                       "Value too large for type!"'
;
; Testcase distilled from the bzip2 SPECint benchmark.
;
; RUN: analyze -exprs %s

implementation

void "sortIt"(ubyte %X)
begin
	%reg115 = shl ubyte %X, ubyte 8
	ret void
end
