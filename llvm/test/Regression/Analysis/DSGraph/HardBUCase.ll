; RUN: llvm-as < %s | opt -analyze -budatastructure -dont-print-ds

%MidFnTy = type void (\2*)

implementation

int %main() {
	call %MidFnTy* %Mid(%MidFnTy* %Mid)
	ret int 0
}

internal void %Mid(%MidFnTy *%F) {
	call void %Bottom(%MidFnTy* %F)
	ret void
}

internal void %Bottom(%MidFnTy* %F) {
	call void %F(%MidFnTy* %Mid)
	ret void
}
