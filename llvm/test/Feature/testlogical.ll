; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

implementation

int "simpleAdd"(int %i0, int %j0)
begin
	%t1 = xor int %i0, %j0
	%t2 = or int %i0, %j0
	%t3 = and int %t1, %t2
	ret int %t3
end

