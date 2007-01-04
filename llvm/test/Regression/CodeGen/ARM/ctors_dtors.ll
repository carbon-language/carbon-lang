; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -o %t.s -f &&
; RUN: grep '\.section \.ctors,"aw",.progbits' %t.s | grep % &&
; RUN: grep '\.section \.dtors,"aw",.progbits' %t.s | grep %

%llvm.global_ctors = appending global [1 x { int, void ()* }] [ { int, void ()* } { int 65535, void ()* %__mf_init } ]		; <[1 x { int, void ()* }]*> [#uses=0]
%llvm.global_dtors = appending global [1 x { int, void ()* }] [ { int, void ()* } { int 65535, void ()* %__mf_fini } ]		; <[1 x { int, void ()* }]*> [#uses=0]

void %__mf_init() {
entry:
	ret void
}

void %__mf_fini() {
entry:
	ret void
}
