; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


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
