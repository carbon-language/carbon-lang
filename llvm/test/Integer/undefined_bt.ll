; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


%X = global i32 undef

implementation

declare i32 "atoi"(i8 *)

define i32 %test() {
	ret i32 undef
}

define i32 %test2() {
	%X = add i32 undef, 1
	ret i32 %X
}
