; RUN: llvm-as < %s | opt -adce | llvm-dis | not grep null

declare int %strlen(sbyte*)

int %test() {
	;; Dead call should be deleted!
	invoke int %strlen(sbyte *null) to label %Cont unwind label %Other
Cont:
	ret int 0
Other:
	ret int 1
}

