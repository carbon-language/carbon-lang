; Uninitialized values are not handled correctly.
;
; RUN: llvm-as < %s | opt -mem2reg
;

implementation

int "test"()
begin
	%X = alloca int           ; To be promoted
	%Y = load int* %X
	ret int %Y
end
