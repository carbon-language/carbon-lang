; RUN: llvm-upgrade < %s | llvm-as | opt -funcresolve

declare int %read(...)

long %read() {
  ret long 0
}

int %testfunc() {
	%X = call int(...)* %read()
	ret int %X
}
