; RUN: not llvm-as -f %s -o /dev/null

; Test that self referential instructions are not allowed

implementation

void "test"()
begin
	%A = add int %A, 0
	ret void
end
