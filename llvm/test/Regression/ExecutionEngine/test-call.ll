; RUN: llvm-as -f %s -o %t.bc
; RUN: lli %t.bc > /dev/null


declare void %exit(int)

int %test(sbyte %C, short %S) {
  %X = cast short %S to ubyte
  %Y = cast ubyte %X to int
  ret int %Y
}

void %FP(void(int) * %F) {
	%X = call int %test(sbyte 123, short 1024)
	call void %F(int %X)
	ret void
}

int %main() {
	call void %FP(void(int)* %exit)
	ret int 1
}
