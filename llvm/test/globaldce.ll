%var = internal global int 1234   ;; used by dead method

implementation

internal int "deadfunc"() {
	%val = load int * %var
	%val2 = call int %deadfunc()
	%val3 = add int %val, %val2
	ret int %val3
}

int "main"(int %argc)   ; TODO: , sbyte **argv, sbyte **envp)
begin
	ret int -1
end

