; RUN: llvm-as < %s | opt -adce | llvm-dis | not grep call

declare int %strlen(sbyte*)

void %test() {
	;; Dead call should be deleted!
	call int %strlen(sbyte *null)
	ret void
}
