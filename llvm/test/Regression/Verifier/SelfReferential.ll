; Test that self referential instructions are not allowed

implementation

void "test"()
begin
	%A = add int %A, 0
	ret void
end
