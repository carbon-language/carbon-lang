; RUN: as < %s | opt -raise | dis | not grep cast

void %test(...) { ret void }

void %caller() {
	call void (...) *%test()
	ret void
}
