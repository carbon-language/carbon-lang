%FunTy = type int(int)

implementation

void "invoke"(%FunTy *%x)
begin
	%foo = call %FunTy* %x(int 123)
	ret void
end

int "main"(int %argc, sbyte **%argv, sbyte **%envp)
begin
        %retval = call int (int) *%test(int %argc)
        %two    = add int %retval, %retval
	%retval2 = call int %test(int %argc)

	%two2 = add int %two, %retval2
	call void %invoke (%FunTy* %test)
        ret int %two2
end

int "test"(int %i0)
begin
    ret int %i0
end
