; RUN: as < %s | opt -funcresolve

declare int %read(...)

long %read(int %fildes, sbyte* %buf, ulong %nbyte) {
  ret long 0
}

int %testfunc() {
	%X = call int(...)* %read()
	ret int %X
}
