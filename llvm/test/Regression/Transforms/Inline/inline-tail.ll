; RUN: llvm-as < %s | opt -inline | llvm-dis | not grep tail

implementation

declare void %bar(int*)

internal void %foo(int* %P) {  ;; to be inlined
  tail call void %bar(int* %P)
  ret void
}

void %caller() {
	%A = alloca int
	call void %foo(int* %A)   ;; not a tail call
	ret void
}
