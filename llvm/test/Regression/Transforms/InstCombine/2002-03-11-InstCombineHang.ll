; This testcase causes instcombine to hang.
;
; RUN: llvm-as < %s | opt -instcombine

implementation

void "test"(int %X)
begin
	%reg117 = add int %X, 0
	ret void
end
