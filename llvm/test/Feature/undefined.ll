
%X = global int undef

implementation

declare int "atoi"(sbyte *)

int %test() {
	ret int undef
}

int %test2() {
	%X = add int undef, 1
	ret int %X
}
