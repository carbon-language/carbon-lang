; Fixed a problem where level raise would try to forward substitute a cast of a
; method pointer type into a call.  In doing so, it would have to change the
; types of the arguments to the call, but broke doing so.
;
; RUN: llvm-as < %s | opt -raise

implementation


void "test"(void (int*) *%Fn, long* %Arg)
begin
	%Fn2 = cast void (int*) *%Fn to void(long*) *
	call void %Fn2(long *%Arg)
	ret void
end
