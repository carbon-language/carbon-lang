; This testcase causes instcombine to hang.
;
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine

implementation

void "test"(int %X)
begin
	%reg117 = add int %X, 0
	ret void
end
