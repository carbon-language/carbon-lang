; RUN: llvm-as < %s | opt -lowersetjmp | llvm-dis | grep invoke

%JmpBuf = type int
%.str_1 = internal constant [13 x sbyte] c"returned %d\0A\00"

implementation

declare void %llvm.longjmp(%JmpBuf *%B, int %Val)
declare int %llvm.setjmp(%JmpBuf *%B)

int %simpletest() {
	%B = alloca %JmpBuf
	%Val = call int %llvm.setjmp(%JmpBuf* %B)
	%V = cast int %Val to bool
	br bool %V, label %LongJumped, label %Normal
Normal:
	call void %llvm.longjmp(%JmpBuf* %B, int 42)
	ret int 0 ;; not reached
LongJumped:
	ret int %Val
}

declare int %printf(sbyte*, ...)

int %main() {
	%V = call int %simpletest()
	call int(sbyte*, ...)* %printf(sbyte* getelementptr ([13 x sbyte]* %.str_1, long 0, long 0), int %V)
	ret int 0
}
