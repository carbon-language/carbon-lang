; RUN: if as < %s | opt -raise | dis | grep cast
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

void %test(...) { ret void }

void %caller() {
	call void (...) *%test()
	ret void
}
