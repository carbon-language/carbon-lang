; RUN: llvm-as < %s | opt -adce

implementation

int "main"(int %argc)
begin
	br label %2

	%retval = phi int [ %argc, %2 ]		; <int>	[#uses=2]
	%two = add int %retval, %retval		; <int>	[#uses=1]
	ret int %two

	br label %1
end

