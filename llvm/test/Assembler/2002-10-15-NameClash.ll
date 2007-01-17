; RUN: llvm-upgrade < %s | llvm-as -o /dev/null -f

declare int "ArrayRef"([100 x int] * %Array)

int "ArrayRef"([100 x int] * %Array) {
	ret int 0
}
