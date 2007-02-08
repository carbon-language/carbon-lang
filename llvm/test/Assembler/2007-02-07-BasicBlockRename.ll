; PR1187
; RUN: llvm-upgrade < %s | llvm-as > /dev/null

implementation

int @main(int %argc, sbyte** %argv) {
entry:
	%exit = alloca int, align 4		; <i32*> [#uses=11]
        store int 0, int* %exit
	br label %exit

exit:
	ret int 0
}
